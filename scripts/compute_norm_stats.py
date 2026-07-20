import json
import numpy as np
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import trange, tqdm
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from lingbotvla.data import build_vla_dataset
from lingbotvla.utils.normalize import (
    RunningStats,
    RunningStatsState,
)
from lingbotvla.models import build_processor
from lingbotvla.utils import helper
from lingbotvla.utils.arguments import parse_args, ModelArguments
from lingbotvla.utils.dist_utils import all_reduce
import lingbotvla.utils.normalize as normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks.vla.train_lingbotvla import MyTrainingArguments, MyDataArguments

logger = helper.create_logger(__name__)


@dataclass
class NormComputeDataArguments(MyDataArguments):
    data_ratio_for_norm_compute: float = field(
        default=1.0,
        metadata={"help": "data ratio for norm compute."},
    )
    robot_name: str = field(
        default=None,
        metadata={"help": "robot name to compute norm."},
    )
    norm_path: str = field(
        default=None,
        metadata={"help": "Path to save norm stats."},
    )
    norm_merge_chunk_dim: bool = field(
        default=True,
        metadata={"help": "If merge chunk dim of action for norm compute."},
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "NormComputeDataArguments" = field(default_factory=NormComputeDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


def get_all_tasks(task_files, robot_name, sep=' '):
    task_files = task_files.split(',')
    data_names, task_list = [], []
    for task_file in task_files:
        assert task_file.lower().endswith('.txt')
        with open(task_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data_name, task = line.split(sep)
                if robot_name is not None and data_name not in robot_name:
                    continue
                data_names.append(data_name)
                task_list.append(task)
        f.close()
    return data_names, task_list


def collate_fn(batch_list):
    if not batch_list:
        return {}
    keys = batch_list[0].keys()
    batch = {}
    for key in keys:
        if isinstance(batch_list[0][key], torch.Tensor):
            batch[key] = torch.stack([item[key] for item in batch_list])
    return batch


def compute_norm(dataset, batch_size, stats, state_norm_keys, action_norm_keys, delta_norm, ratio,
                 rank=0, world_size=1, num_workers=8, norm_merge_chunk_dim=False):
    if ratio < 1:
        num_step = int(len(dataset)*ratio)
        random.seed(42)
        data_ids = random.sample(range(len(dataset)), num_step)
    else:
        data_ids = list(range(len(dataset)))

    data_ids = data_ids[rank::world_size]

    subset_dataset = Subset(dataset, data_ids)

    data_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=False,
        multiprocessing_context='fork',
        drop_last=False,
    )

    total_batches = len(data_loader)

    pbar = tqdm(
        data_loader,
        total=total_batches,
        unit="batch",
        ncols=100,
        disable=(rank != 0),
        desc=f"rank{rank}",
    )
    for batch in pbar:
        for key in state_norm_keys:
            if key in batch:
                values = np.asarray(batch[key])
                stats[key].update(values.reshape(-1, values.shape[-1]))
        for key in action_norm_keys:
            if key in batch:
                values = np.asarray(batch[key]) if (not delta_norm.get(key, False) or norm_merge_chunk_dim) else np.asarray(batch[key].reshape(batch[key].shape[0], -1))
                stats[key].update(values.reshape(-1, values.shape[-1]))

    del data_loader
    del subset_dataset
    del dataset


def get_norm_stats(stats, delta_norm, chunk_size, norm_merge_chunk_dim=False):
    assert stats is not None
    norm_stats = {}
    for key, state in stats.items():
        _chunk_size = chunk_size if (key in delta_norm and delta_norm[key]==True) and not norm_merge_chunk_dim else None
        norm_stats[key] = state.get_statistics(chunk_size=_chunk_size)
    return norm_stats


def _init_dataset_worker(args, data_names):
    args.data.chunk_size = args.train.chunk_size
    dataset = build_vla_dataset(dataset_config=args.data, 
                                model_config=None, 
                                config=None, 
                                processor=None, 
                                do_nomalize=False,
                                return_item=True,
                                disabled_image_features=True,
                                load_only_actions_and_states=True)
    return dataset


if __name__ == "__main__":
    args = parse_args(Arguments)

    if args.train.world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(f"cuda:{args.train.local_rank}")
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank = args.train.global_rank
    world_size = args.train.world_size

    logger.info(f"Process rank: {rank}, world size: {world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))

    logger.info_rank0("Prepare data")
    stats = None

    assert args.data.datasets_type == 'vla'

    robot_name = args.data.robot_name.split(',') if args.data.robot_name is not None else None
    
    if args.data.data_name == 'multi':
        data_names, repo_ids = get_all_tasks(args.data.train_path, robot_name)
        if robot_name is None:
            assert len(set(data_names)) == 1
        else:
            for data_name in set(data_names):
                assert data_name in robot_name
    else:
        data_names, repo_ids = [args.data.data_name], [args.data.train_path]
        args.data.data_name = 'multi'

    filename = '_'.join(list(set(data_names)))
    tmp_dir = f"tmp/"
    if rank == 0:
        os.makedirs(tmp_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    filename = os.path.join(tmp_dir, f"tmp_{filename}_rank{rank}.txt")
    with open(filename, 'w') as f:
        for robot, task in zip(data_names, repo_ids):
            f.write(f"{robot} {task}\n")
    f.close()
    args.data.train_path = filename
    dataset = _init_dataset_worker(args, data_names)
    if rank == 0:
        print(f"===========\nProcessing {len(dataset._datasets)} lerobot datasets\n===========")
    os.remove(filename)
    assert len(list(set([' '.join(_datasets.state_features+_datasets.action_features) for _datasets in dataset._datasets])))==1

    start = time.time()
    iter_times = []
    for i, data in enumerate(dataset):
        end = time.time()
        iter_times.append(end - start)
        logger.info(f"Iter {i} time: {iter_times[-1]:.6f}s")
        start = time.time()
        if i >= 1:
            break
    if len(iter_times) >= 2:
        logger.info(f"Total time for 2 iterations: {sum(iter_times):.6f}s, Average: {sum(iter_times)/2:.6f}s")

    state_norm_keys = dataset._datasets[0].state_features
    action_norm_keys = dataset._datasets[0].action_features
    delta_norm = dataset._datasets[0].feature_transform.action_subtract_state
    stats = {key: normalize.RunningStats() for key in action_norm_keys+state_norm_keys}
    chunk_size = args.data.chunk_size
    
    ratio = args.data.data_ratio_for_norm_compute
    print(f"Start computing norm stats with ratio={ratio}, num_workers={args.data.num_workers}")
    compute_norm(dataset._datasets[0].dataset, args.train.micro_batch_size, stats, state_norm_keys, action_norm_keys,
                 delta_norm, ratio=ratio, rank=rank, world_size=world_size,
                 num_workers=args.data.num_workers, norm_merge_chunk_dim=args.data.norm_merge_chunk_dim)
    print(f"End computing norm stats computed with ratio={ratio}")

    if world_size > 1:
        local_state = {
            k: (v.get_state().model_dump() if v._count > 0 else None)
            for k, v in stats.items()
        }
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_state)
        if rank == 0:
            merged = {}
            for key in stats.keys():
                objs = []
                for shard in gathered:
                    if shard is None or shard.get(key) is None:
                        continue
                    objs.append(RunningStats.from_state(RunningStatsState(**shard[key])))
                if not objs:
                    raise RuntimeError(f"No rank produced any data for key={key!r}")
                merged[key] = RunningStats.merge(objs)
            stats = merged
        dist.barrier()

    if rank == 0:
        norm_stats = get_norm_stats(stats, delta_norm, chunk_size, args.data.norm_merge_chunk_dim)
        output_path = Path(args.data.norm_path)
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats, stats[state_norm_keys[0]]._count)

    if world_size > 1:
        dist.destroy_process_group()