import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import websockets.sync.client

from typing_extensions import override

subprocess.run(
    ["sudo", "rm", "-rf", "/dev/shm/zcm"],
    capture_output=True,
    text=True,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.real_env_sdk import make_real_env
from openpi_client import image_tools
from robot_infer.msgpack_numpy import Packer, unpackb

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    LEROBOT_DATASET_API = "v3"
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    LEROBOT_DATASET_API = "v2"

from lingbotvla.data.vla_data.base_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

DT = 1 / 30
DEFAULT_PROMPT = "clear the bin box"


def camera_aliases(camera_name: str) -> list[str]:
    return [
        camera_name,
        f"rs/{camera_name}",
        f"rs.{camera_name}",
        f"rs_{camera_name}",
    ]


def get_camera_image(images: dict, camera_name: str):
    for alias in camera_aliases(camera_name):
        if alias in images:
            return images[alias]

    rs_images = images.get("rs")
    if isinstance(rs_images, dict) and camera_name in rs_images:
        return rs_images[camera_name]

    raise KeyError(f"Camera '{camera_name}' not found in observation images.")


class WebsocketInferenceClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 55555,
        api_key=None
    ) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self):
        return self._server_metadata

    def _wait_for_server(self):
        logger.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = unpackb(conn.recv())
                logger.info(f"Connected to server. Metadata: {metadata}")
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs):
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    @override
    def reset(self, robo_name: str) -> None:
        self.infer(dict(reset=True, robo_name=robo_name))

    def close(self) -> None:
        if hasattr(self, '_ws') and self._ws is not None:
            self._ws.close()

    def compress_image(self, image, depth=False):
        if depth:
            return image
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return buffer

    def close(self):
        if hasattr(self, '_ws') and self._ws is not None:
            self._ws.close()
            logger.info("Connection closed")


class ActionSmooth:
    def __init__(self, client, max_timesteps: int) -> None:
        self.action_horizon = 30
        self.base_delay = 0
        self.query_frequency = 30
        self.all_time_actions = np.zeros(
            [max_timesteps, max_timesteps + self.action_horizon - self.base_delay, 23],
            dtype=np.float32,
        )
        self.action_keep = None
        self.t = 0
        self.client = client

    def get_action(self, observation):
        if self.t % self.query_frequency == 0:
            self.action_keep = self.client.infer(observation)["actions"][self.base_delay:self.action_horizon, ...]
            self.all_time_actions[self.t, self.t:self.t + self.action_horizon - self.base_delay] = self.action_keep

        actions_for_curr_step = self.all_time_actions[:, self.t]
        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]

        if len(actions_for_curr_step) == 0:
            return np.zeros(23, dtype=np.float32)

        base_action = actions_for_curr_step[-1, 19:]

        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / np.sum(exp_weights)
        exp_weights = exp_weights[:, np.newaxis]
        action = np.sum(actions_for_curr_step * exp_weights, axis=0, keepdims=True)
        action = action.squeeze(0)

        self.t += 1
        return np.concatenate([action[:19], base_action])


def prepare_observation(observation, client: WebsocketInferenceClient, camera_names: list[str], prompt: str):
    observation["observation.state"] = observation["qpos"]
    observation["task"] = prompt

    for camera_name in camera_names:
        image = get_camera_image(observation["images"], camera_name)
        observation[f"observation.images.{camera_name}"] = image

    return observation


def warm_up(action_smooth, observation, client: WebsocketInferenceClient, args):
    logger.info("Warm up")
    observation = prepare_observation(observation, client, args.camera_names, "")

    for _ in range(args.warmup_steps):
        action_smooth.client.infer(observation)


def load_hdf5(ep_path):
    with h5py.File(ep_path, "r") as ep:
        state_arm = ep["/observation/state/arm/position"][:]
        state_effector = ep["/observation/state/effector/position"][:]
        state_waist = ep["/observation/state/waist/position"][:]
        state_head = ep["/observation/state/head/position"][:]
        state_base = ep["/observation/state/base/velocity"][:]

        state = np.concatenate(
            [
                state_arm[:, :7],
                state_effector[:, :-1],
                state_arm[:, 7:],
                state_effector[:, -1:],
                state_waist[:],
                state_head[:],
                state_base[:],
            ],
            axis=1,
        )
        action = state
        prompt = ep.attrs["task_name"]
        logger.info("Loaded task prompt: %s", prompt)
    return action


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def discover_feature_keys(dataset_item):
    image_keys = []
    state_keys = []
    action_keys = []
    for key in dataset_item.keys():
        if "images" in key:
            image_keys.append(key)
        elif "state" in key and "images" not in key:
            state_keys.append(key)
        elif "action" in key:
            action_keys.append(key)
    return image_keys, state_keys, action_keys


def prepare_inference_observation(dataset_item, image_keys, state_keys):
    obs = {}
    for image_key in image_keys:
        image = dataset_item[image_key]
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()
        obs[image_key] = image.astype(np.uint8)

    for state_key in state_keys:
        state = dataset_item[state_key]
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        obs[state_key] = state

    if "task" in dataset_item:
        obs["task"] = dataset_item["task"]

    return obs


def open_loop_eval_main(args):
    from scripts.open_loop_eval import plot_trajectory_results

    logger.info("Starting open-loop evaluation via WebSocket...")

    client = WebsocketInferenceClient(host=args.host, port=args.port)
    logger.info(f"Connected to server at {args.host}:{args.port}")

    data_path = Path(args.data_path)
    if data_path.is_absolute() and data_path.exists():
        repo_id = data_path.name
        root = data_path
    else:
        repo_id = args.data_path
        root = None

    logger.info(f"Loading dataset: repo_id={repo_id}, root={root}")
    dataset_meta = LeRobotDatasetMetadata(repo_id, root=root)

    delta_timestamps = {}
    sample_item = None
    for _ in range(min(10, len(dataset_meta.episodes))):
        try:
            temp_dataset = LeRobotDataset(repo_id, root=root, delta_timestamps={})
            sample_item = temp_dataset[0]
            break
        except Exception:
            continue

    if sample_item is None:
        raise RuntimeError("Failed to load sample item from dataset")

    image_keys, state_keys, action_keys = discover_feature_keys(sample_item)
    logger.info(f"Discovered keys - images: {image_keys}, states: {state_keys}, actions: {action_keys}")

    delta_timestamps = {}
    for action_feature in action_keys:
        delta_timestamps[action_feature] = [t / dataset_meta.fps for t in range(args.chunk_size)]

    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
    logger.info(f"Dataset length: {len(dataset)}")

    all_mse = []
    all_mae = []

    for traj_id in args.traj_ids:
        if LEROBOT_DATASET_API == "v2":
            valid_episode_ids = dataset.meta.episodes.keys()
        else:
            valid_episode_ids = dataset.meta.episodes["episode_index"]

        if traj_id not in valid_episode_ids:
            logger.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logger.info(f"Running trajectory: {traj_id}")

        reset_msg = {"reset": True, "robo_name": args.robo_name}
        client.infer(reset_msg)
        logger.info("Sent reset message to server")

        if LEROBOT_DATASET_API == "v2":
            start_id, end_id = dataset.episode_data_index['from'][traj_id], dataset.episode_data_index['to'][traj_id]
        else:
            start_id, end_id = dataset.meta.episodes[traj_id]["dataset_from_index"], dataset.meta.episodes[traj_id]["dataset_to_index"]

        gt_action_across_time = []
        state_joints_across_time = []
        pred_action_across_time = []

        count = 0
        for data_id in range(start_id, end_id, args.chunk_size):
            if count >= args.max_infer_time:
                break

            try:
                dataset_item = dataset[data_id]
            except IndexError:
                break

            obs = prepare_inference_observation(dataset_item, image_keys, state_keys)

            gt_action_chunk = []
            for action_key in action_keys:
                action_val = dataset_item[action_key]
                if isinstance(action_val, torch.Tensor):
                    action_val = action_val.cpu().numpy()
                gt_action_chunk.append(action_val[:args.chunk_size])
            gt_action_chunk = np.concatenate(gt_action_chunk, axis=-1)
            gt_action_across_time.append(gt_action_chunk)

            state_chunk = []
            for state_key in state_keys:
                state_val = dataset_item[state_key]
                if isinstance(state_val, torch.Tensor):
                    state_val = state_val.cpu().numpy()
                state_chunk.append(state_val.reshape(1, -1))
            state_chunk = np.concatenate(state_chunk, axis=-1)
            state_joints_across_time.append(state_chunk)

            preds = client.infer(obs)
            if "server_timing" in preds:
                del preds["server_timing"]

            pred_action_chunk = []
            for action_key in action_keys:
                if action_key in preds:
                    pred_val = preds[action_key]
                    if isinstance(pred_val, torch.Tensor):
                        pred_val = pred_val.cpu().numpy()
                    pred_action_chunk.append(pred_val)
            if pred_action_chunk:
                pred_action_chunk = np.concatenate(pred_action_chunk, axis=-1)
                pred_action_across_time.append(pred_action_chunk)

            count += 1
            logger.info(f"Trajectory {traj_id}: step {count}/{args.max_infer_time}, data_id {data_id}")

        if not gt_action_across_time or not pred_action_across_time:
            logger.warning(f"No data for trajectory {traj_id}")
            continue

        gt_action_across_time = np.concatenate(gt_action_across_time, axis=0)
        state_joints_across_time = np.concatenate(state_joints_across_time, axis=0)
        pred_action_across_time = np.concatenate(pred_action_across_time, axis=0)

        min_len = min(gt_action_across_time.shape[0], pred_action_across_time.shape[0])
        gt_action_across_time = gt_action_across_time[:min_len]
        pred_action_across_time = pred_action_across_time[:min_len]

        mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
        mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))

        logger.info(f"Trajectory {traj_id} - MSE: {mse}, MAE: {mae}")
        logger.info(f"gt_action shape: {gt_action_across_time.shape}, pred_action shape: {pred_action_across_time.shape}")

        all_mse.append(mse)
        all_mae.append(mae)

        save_plot_path = os.path.join(args.save_plot_path, f'{traj_id}.png')
        os.makedirs(args.save_plot_path, exist_ok=True)

        plot_trajectory_results(
            state_joints_across_time=state_joints_across_time[:min_len],
            gt_action_across_time=gt_action_across_time,
            pred_action_across_time=pred_action_across_time,
            traj_id=traj_id,
            action_keys=action_keys,
            action_horizon=args.chunk_size,
            save_plot_path=save_plot_path,
        )

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logger.info(f"Average MSE across all trajs: {avg_mse}")
        logger.info(f"Average MAE across all trajs: {avg_mae}")
        print(f"Average MSE across all trajs: {avg_mse}")
        print(f"Average MAE across all trajs: {avg_mae}")
    else:
        logger.info("No valid trajectories were evaluated.")

    client.close()
    logger.info("Open-loop evaluation completed")


def main(args):
    env = make_real_env(camera_names=args.camera_names)

    time.sleep(2)
    env.move_to_init_pose()

    client = WebsocketInferenceClient(host=args.host, port=args.port)
    action_smooth = ActionSmooth(client, max_timesteps=args.num_steps)

    reset_msg = {
        'reset': True,
        'robo_name': 'zerith',
    }
    logger.info("Sending reset message to initialize policy...")
    client.infer(reset_msg)

    observation = env.reset().observation
    observation["state"] = observation["qpos"]
    warm_up(action_smooth, observation, client, args)

    data_action = None
    if args.init_hdf5:
        data = load_hdf5(args.init_hdf5)
        data_action = data[args.init_frame_idx]
        env.move_to_target_joint(data_action[:-2])
        time.sleep(4)
        logger.info("Loaded initialization action: %s", data_action)

    if not args.skip_pause:
        logger.info("Paused before inference. Adjust the robot, then continue to start policy control.")
        import pdb
        pdb.set_trace()

    observation = env.reset().observation
    observation = env.get_observation().observation

    try:
        for step in range(args.num_steps):
            observation["state"] = observation["qpos"]

            time0 = time.time()
            observation = prepare_observation(observation, client, args.camera_names, args.prompt)
            action = np.copy(action_smooth.get_action(observation))
            observation = env.get_observation().observation

            if data_action is not None:
                action[-4:-2] = data_action[-4:-2]

            if step % 10 == 0:
                logger.info(f"Step {step}: action = {action}")

            env.step_joint(action[:-2])
            elapsed_time = time.time() - time0
            time.sleep(max(0, DT - elapsed_time))

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        client.close()
        logger.info("Inference completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zerith Inference WebSocket Client")

    parser.add_argument("--host", type=str, default="127.0.0.1", help="Policy server host")
    parser.add_argument("--port", type=int, default=55555, help="Policy server port")
    parser.add_argument("--api_key", type=str, default=None, help="API key for authentication")

    parser.add_argument("--open_loop_eval", action="store_true", help="Enable open-loop evaluation mode")
    parser.add_argument("--data_path", type=str, default=None, help="Path to validation data for open-loop eval")
    parser.add_argument("--robo_name", type=str, default="zerith", help="Robot config name")
    parser.add_argument("--traj_ids", type=int, nargs='+', default=[0], help="Trajectory IDs to evaluate")
    parser.add_argument("--chunk_size", type=int, default=50, help="Chunk size for evaluation")
    parser.add_argument("--max_infer_time", type=int, default=10, help="Max number of inference calls")
    parser.add_argument("--save_plot_path", type=str, default='./open_loop_test/', help="Path to save plots")

    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Language instruction")
    parser.add_argument("--num_steps", type=int, default=20000, help="Number of control steps")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup inference calls")
    parser.add_argument("--init_hdf5", type=str, default=None, help="Optional HDF5 file used for initialization")
    parser.add_argument("--init_frame_idx", type=int, default=30, help="Frame index used from the initialization HDF5")
    parser.add_argument(
        "--camera_names",
        nargs="+",
        type=str,
        choices=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        help="Camera names",
    )
    parser.add_argument("--skip_pause", action="store_true", help="Skip the pause before starting inference")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    args = parser.parse_args()

    if args.open_loop_eval:
        if not args.data_path:
            parser.error("--data_path is required for open-loop evaluation")
        open_loop_eval_main(args)
    else:
        main(args)