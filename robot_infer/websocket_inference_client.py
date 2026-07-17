import argparse
import logging
import os
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np
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

logger = logging.getLogger(__name__)

DT = 1 / 30
DEFAULT_PROMPT = "grab all the ducks and put them into the basket"


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
    def reset(self) -> None:
        pass

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
    observation["state"] = observation["qpos"]
    observation["prompt"] = prompt

    for camera_name in camera_names:
        image = get_camera_image(observation["images"], camera_name)
        observation["images"][camera_name] = client.compress_image(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

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


def main(args):
    env = make_real_env(camera_names=args.camera_names)

    time.sleep(2)
    env.move_to_init_pose()

    client = WebsocketInferenceClient(host=args.host, port=args.port)
    action_smooth = ActionSmooth(client, max_timesteps=args.num_steps)

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

            if action[15] > 0.7:
                action[15] = 1.3

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
    parser.add_argument("--api_key", type=str, default=None, help="API key for authentication")
    parser.add_argument("--skip_pause", action="store_true", help="Skip the pause before starting inference")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    args = parser.parse_args()
    main(args)