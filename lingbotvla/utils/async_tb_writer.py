"""Non-blocking WandB writer.

Directly logs to wandb instead of using TensorBoard queue.
"""

import logging

import numpy as np
import torch
import wandb

logger = logging.getLogger(__name__)


class AsyncTBWriter:

    def __init__(self, log_dir: str = None):
        pass

    def add_scalar(self, tag, scalar_value, global_step):
        wandb.log({tag: scalar_value}, step=global_step)

    def add_histogram(self, tag, values, global_step):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        wandb.log({tag: wandb.Histogram(values)}, step=global_step)

    def add_histogram_from_counts(self, tag, counts_cpu, global_step):
        total = counts_cpu.sum().item()
        if total <= 0:
            return
        max_hist_events = 100000
        if total > max_hist_events:
            counts_cpu = counts_cpu * (max_hist_events / total)
        counts_long = counts_cpu.round().long().clamp(min=0)
        num_experts = counts_long.numel()
        indices = torch.repeat_interleave(
            torch.arange(num_experts), counts_long
        )
        if indices.numel() > 0:
            indices_np = indices.numpy()
            wandb.log({tag: wandb.Histogram(indices_np)}, step=global_step)

    def add_expert_bar(self, tag, counts_cpu, global_step, title=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        counts = counts_cpu.detach().float().numpy()
        n = counts.shape[0]
        fig = plt.figure(figsize=(10, 4))
        plt.bar(range(n), counts)
        avg = counts.mean()
        plt.axhline(avg, color="r", linestyle="--", linewidth=1, label=f"avg={avg:.0f}")
        plt.title(title or f"Expert load (step {global_step})")
        plt.xlabel("Expert ID")
        plt.ylabel("Token count")
        plt.legend(loc="upper right", fontsize=8)
        wandb.log({tag: fig}, step=global_step)
        plt.close(fig)

    def flush(self):
        wandb.flush()

    def close(self):
        self.flush()
