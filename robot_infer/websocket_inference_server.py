import argparse
import asyncio
import http
import logging
import os
import sys
import time
import traceback
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import websockets.asyncio.server as _server
import websockets.frames
import yaml
import numpy as np
import torch
from torchvision.transforms.v2 import Resize
from glob import glob
from tqdm import tqdm
from safetensors import safe_open

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import AutoConfig

from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
from lingbotvla.data.vla_data.utils import FeatureTransform
from lingbotvla.models import build_processor

logger = logging.getLogger(__name__)


BASE_MODEL_PATH = {
    'qwen3vl': os.environ.get(
        'QWEN3VL_PATH',
        'Qwen/Qwen3-VL-4B-Instruct/',
    ),
}


def set_seed_everywhere(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class PolicyPreprocessMixin:
    @staticmethod
    def _to_device_image_grid_thw(image_grid_thw, device):
        if image_grid_thw is None:
            return None
        return image_grid_thw.to(device=device, dtype=torch.long)

    @torch.no_grad
    def select_action(
        self, observation: dict[str, torch.Tensor], use_bf16: bool = False
    ):
        self.eval()
        device = 'cuda'
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        if len(observation['images'].shape) == 4:
            observation['images'] = observation['images'].unsqueeze(0)
            observation['img_masks'] = observation['img_masks'].unsqueeze(0)

        actions = self.model.sample_actions(
            observation['images'].to(dtype=dtype, device=device),
            observation['img_masks'].to(device=device),
            observation['lang_tokens'].unsqueeze(0).to(device=device),
            observation['lang_masks'].unsqueeze(0).to(device=device),
            observation['state'].unsqueeze(0).to(dtype=dtype, device=device),
            image_grid_thw=self._to_device_image_grid_thw(observation.get('image_grid_thw'), device),
        )
        observation['actions'] = actions.squeeze(0).to(dtype=torch.float32, device='cpu')
        if use_bf16:
            observation['state'] = observation['state'].to(dtype=torch.float32)
        data = self.feature_transform.unapply(observation)
        return data

    @torch.no_grad
    def sample_actions_batch(
        self,
        observation: dict[str, torch.Tensor],
        use_bf16: bool = False,
        use_compile: bool = False,
        capture_time: bool = False,
        sample_compile_fn=None,
    ) -> torch.Tensor:
        self.eval()
        device = "cuda"
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        images = observation["images"]
        img_masks = observation["img_masks"]
        lang_tokens = observation["lang_tokens"]
        lang_masks = observation["lang_masks"]
        state = observation["state"]
        image_grid_thw = observation.get("image_grid_thw", None)

        has_batch_dim = img_masks.ndim >= 2
        if not has_batch_dim:
            images = images.unsqueeze(0)
            img_masks = img_masks.unsqueeze(0)
        if lang_tokens.ndim == 1:
            lang_tokens = lang_tokens.unsqueeze(0)
            lang_masks = lang_masks.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        actions = sample_compile_fn(
            images.to(dtype=dtype, device=device),
            img_masks.to(device=device),
            lang_tokens.to(device=device),
            lang_masks.to(device=device),
            state.to(dtype=dtype, device=device),
            image_grid_thw=self._to_device_image_grid_thw(image_grid_thw, device),
        )

        return actions.to(dtype=torch.float32, device="cpu")


class LingBotVlaV2InferencePolicy(PolicyPreprocessMixin, LingbotVlaV2Policy):
    pass


class LingbotVLAv2InferenceServer:
    def __init__(
        self,
        path_to_pi_model="",
        base_model_path=None,
        robot_norm_path=None,
        use_length=1,
        chunk_ret=True,
        use_bf16=True,
        use_fp32=False,
        use_compile=False,
    ) -> None:
        assert not (use_bf16 and use_fp32), 'Bfloat16 or Float32!!!'
        self.use_length = use_length
        self.chunk_ret = chunk_ret
        self.robot_norm_path = robot_norm_path
        self.use_compile = use_compile
        self.use_bf16 = use_bf16
        self.use_fp32 = use_fp32

        apply_lingbot_qwen3_vl_patch()

        self.vla = self.load_vla(path_to_pi_model, base_model_path)
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16).cuda().eval()
        else:
            self.vla.model.float()
            self.vla = self.vla.cuda().eval()

        self.global_step = 0
        self.last_action_chunk = None
        self.last_normalized_action_chunk = None
        self.action_key: str = "action"

    def load_model_weights(self, path_to_pi_model, strict=True):
        all_safetensors = glob(os.path.join(path_to_pi_model, "*.safetensors"))
        merged_weights = {}
        for file_path in tqdm(all_safetensors):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged_weights[key] = f.get_tensor(key)
        self.vla.load_state_dict(merged_weights, strict=strict)

    def merge_qwen_config(self, qwen_config):
        if hasattr(qwen_config, 'to_dict'):
            config_dict = qwen_config.to_dict()
        else:
            config_dict = qwen_config

        text_keys = {
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "rms_norm_eps",
            "rope_theta", "vocab_size", "max_position_embeddings",
            "hidden_act", "tie_word_embeddings", "tokenizer_path",
        }

        text_config = config_dict.get("text_config", {})
        for key in text_keys:
            if key in text_config:
                setattr(self.config, key, text_config[key])
            elif key in config_dict:
                setattr(self.config, key, config_dict[key])

        if "vision_config" in config_dict:
            self.config.vision_config = qwen_config.vision_config

    def load_vla(self, path_to_pi_model, base_model_path=None) -> LingbotVlaV2Policy:
        logger.info(f"Loading model from: {path_to_pi_model}")

        training_config_path = Path(path_to_pi_model).parent.parent.parent / 'lingbotvla_cli.yaml'
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)

        training_model_config = training_config['model']
        training_model_config.update(training_config['train'])
        config = LingbotVLAV2Config(**training_model_config)
        for key, value in training_model_config.items():
            if not hasattr(config, key):
                setattr(config, key, value)

        config.attention_implementation = 'eager'

        training_base_model = training_config['model']['tokenizer_path']
        if 'qwen3' in training_base_model.lower() and 'vl' in training_base_model.lower():
            model_name = 'qwen3vl'
        else:
            raise ValueError(f"Unsupported base model: {path_to_pi_model}")
        qwen_base_model_path = os.environ.get('QWEN3VL_PATH', training_base_model) or BASE_MODEL_PATH[model_name]
        config.tokenizer_path = qwen_base_model_path
        self.model_name = model_name

        self.config = config
        qwen_config = AutoConfig.from_pretrained(qwen_base_model_path)
        self.merge_qwen_config(qwen_config)
        config = self.config

        if 'vocab_size' in training_config['model'] and training_config['model']['vocab_size'] != 0:
            config.vocab_size = training_config['model']['vocab_size']
        config.use_cache = True

        self.processor = build_processor(qwen_base_model_path)
        self.language_tokenizer = self.processor.tokenizer
        data_config = SimpleNamespace(**training_config['data'])

        logger.info('Initializing model ... ')
        self.vla = LingBotVlaV2InferencePolicy(config, eval=True)

        if base_model_path is not None:
            logger.info(f'Loading base model weights from: {base_model_path}')
            self.load_model_weights(base_model_path, strict=True)

        use_lora = training_config['train'].get('use_lora', False)
        if use_lora:
            logger.info('Injecting LoRA adapters...')
            from lingbotvla.utils.lora_utils import add_lora_to_model
            add_lora_to_model(
                self.vla.model,
                lora_rank=training_config['train'].get('lora_rank', 4),
                lora_alpha=training_config['train'].get('lora_alpha', 4),
                lora_target_modules=training_config['train'].get('lora_target_modules', 'q,k,v,o,ffn.0,ffn.2'),
                pretrained_lora_path=None,
                lora_target_modules_support=training_config['train'].get('lora_target_modules_support', 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj').split(','),
            )

        logger.info(f'Loading LoRA model weights from: {path_to_pi_model}')
        self.load_model_weights(path_to_pi_model, strict=False)

        self.vla.feature_transform = None
        self.data_config = data_config
        self.config = config
        self.vla.model._use_compile_predict_velocity = bool(self.use_compile)
        self.vla.model._compiled_predict_velocity = None
        self.sample_actions_fn = self.vla.model.sample_actions
        if self.use_compile:
            self.vla.model.qwenvl_with_expert = torch.compile(self.vla.model.qwenvl_with_expert)
            self.sample_actions_fn = torch.compile(self.vla.model.sample_actions)

        if self.robot_norm_path is None:
            self.robot_norm_path = data_config.norm_stats_file

        logger.info('Model initialized ... ')
        return self.vla

    def reset(self, robo_name, path_to_pi_model=None) -> None:
        if path_to_pi_model is not None:
            self.vla = self.load_vla(path_to_pi_model)
            if self.use_bf16:
                self.vla = self.vla.to(torch.bfloat16).cuda().eval()
            else:
                self.vla.model.float()
                self.vla = self.vla.cuda().eval()

        self.global_step = 0
        self.last_action_chunk = None
        self.last_normalized_action_chunk = None

        robot_config = f'configs/robot_configs/{robo_name}.yaml'
        with open(robot_config, 'r') as f:
            self.robot_config = yaml.safe_load(f)

        feature_transform = FeatureTransform(
            robot_config, self.data_config, self.config, self.processor,
            chunk_size=self.config.chunk_size, norm_stats_path=self.robot_norm_path
        )
        self.vla.feature_transform = feature_transform
        self.action_key = feature_transform.org_features["actions"]

    def resize_image(self, observation):
        image_features = self.vla.feature_transform.org_features['images']
        image_size = getattr(self.data_config, 'img_size', 256)
        resize = Resize((image_size, image_size))
        for image_feature in image_features:
            assert image_feature in observation
            assert len(observation[image_feature].shape) == 3 and observation[image_feature].shape[-1] == 3
            image = torch.as_tensor(observation[image_feature]).permute(2, 0, 1).contiguous()
            image = image.to(dtype=torch.float32)
            observation[image_feature] = resize(image)

    def _unapply_batched_actions(self, transformed_observations, actions):
        action_chunk = {}
        for action in self.action_key:
            action_chunk[action] = []

        for transformed, action in zip(transformed_observations, actions):
            single = dict(transformed)
            single['actions'] = action.to(dtype=torch.float32, device='cpu')
            if self.use_bf16 and 'state' in single:
                single['state'] = single['state'].to(dtype=torch.float32)
            data = self.vla.feature_transform.unapply(single)

            for action in self.action_key:
                value = data[action]
                if isinstance(value, torch.Tensor):
                    value = value.float().cpu().numpy()
                else:
                    value = np.asarray(value, dtype=np.float32)
                action_chunk[action].append(value)

        for action_key in action_chunk.keys():
            action_chunk[action_key] = np.stack(action_chunk[action_key], axis=0)

        return action_chunk

    def _prepare_model_input(self, observation):
        observation = dict(observation)
        self.resize_image(observation)
        for k, v in list(observation.items()):
            if isinstance(v, np.ndarray):
                observation[k] = torch.from_numpy(v)
        observation = self.vla.feature_transform.apply(observation, policy_eval=True)
        if self.use_bf16:
            observation['state'] = observation['state'].to(torch.bfloat16)
        return observation

    @staticmethod
    def _pad_and_stack_tensors(values):
        shapes = [tuple(value.shape) for value in values]
        if len(set(shapes)) == 1:
            return torch.stack(values, dim=0)

        if all(value.ndim == 1 for value in values):
            max_len = max(value.shape[0] for value in values)
            fill_value = False if values[0].dtype == torch.bool else 0
            padded = []
            for value in values:
                out = torch.full(
                    (max_len,), fill_value, dtype=value.dtype, device=value.device
                )
                out[:value.shape[0]] = value
                padded.append(out)
            return torch.stack(padded, dim=0)

        raise ValueError(f"Cannot batch tensors with different shapes: {shapes}")

    def _infer_batch(self, observations, return_normalized=False):
        if not isinstance(observations, (list, tuple)) or len(observations) == 0:
            raise ValueError("batch observation must be a non-empty list")
        applied = [self._prepare_model_input(obs) for obs in observations]
        batch_observation = {}
        for key in applied[0].keys():
            values = [item[key] for item in applied]
            if isinstance(values[0], torch.Tensor):
                batch_observation[key] = self._pad_and_stack_tensors(values)
            else:
                batch_observation[key] = values

        actions = self.vla.sample_actions_batch(
            batch_observation,
            self.use_bf16,
            self.use_compile,
            capture_time=False,
            sample_compile_fn=self.sample_actions_fn,
        )

        unnormalized_actions = self._unapply_batched_actions(applied, actions)
        if return_normalized:
            return unnormalized_actions, actions
        return unnormalized_actions

    def infer(self, observation, return_normalized=False):
        if 'reset' in observation and observation['reset']:
            self.reset(
                robo_name=observation['robo_name'],
                path_to_pi_model=observation['path_to_pi_model'] if 'path_to_pi_model' in observation else None
            )
            return dict(action=None)

        is_batch = 'batch' in observation
        observations = observation['batch'] if is_batch else [observation]

        should_forward = (
            self.chunk_ret
            or self.last_action_chunk is None
            or (return_normalized and self.last_normalized_action_chunk is None)
            or self.global_step % self.use_length == 0
            or self.use_length == -1
        )

        if should_forward:
            if return_normalized:
                unnormalized_actions, normalized_actions = self._infer_batch(
                    observations, return_normalized=True,
                )
            else:
                unnormalized_actions = self._infer_batch(observations)
                normalized_actions = None

            if self.use_length > 0:
                for output_key in unnormalized_actions.keys():
                    assert self.use_length <= unnormalized_actions[output_key].shape[1]
                    unnormalized_actions[output_key] = unnormalized_actions[output_key][:, :self.use_length]
                if normalized_actions is not None:
                    assert self.use_length <= normalized_actions.shape[1]
                    normalized_actions = normalized_actions[:, :self.use_length]

            self.last_action_chunk = unnormalized_actions
            self.last_normalized_action_chunk = normalized_actions

        if self.chunk_ret:
            action = self.last_action_chunk
            normalized_action = self.last_normalized_action_chunk
        else:
            step_idx = self.global_step % self.use_length
            action = {}
            for action_key in self.last_action_chunk.keys():
                action[action_key] = self.last_action_chunk[action_key][:, step_idx]
            normalized_action = (
                self.last_normalized_action_chunk[:, step_idx]
                if self.last_normalized_action_chunk is not None
                else None
            )

        if not is_batch:
            for action_key in action:
                action[action_key] = action[action_key][0]
            if normalized_action is not None:
                normalized_action = normalized_action[0]

        result = action
        if return_normalized:
            result = dict(result)
            result["_normalized_actions"] = normalized_action

        self.global_step += 1
        return result


class WebsocketInferenceServer:
    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 55555,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=_optional_float_env("WEBSOCKET_PING_INTERVAL"),
            ping_timeout=_optional_float_env("WEBSOCKET_PING_TIMEOUT"),
            process_request=_health_check,
        ) as server:
            logger.info(f"Inference server started on {self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")

        from robot_infer.msgpack_numpy import Packer, unpackb
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.lower() == "none":
        return None
    return float(value)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Launch the Zerith Inference WebSocket Server")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weights"
    )

    parser.add_argument(
        "--robot_norm_path",
        type=str,
        default=None,
        help="Path to robot normalization stats file"
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to the base pretrained model (lingbot-vla-v2-6b)"
    )

    parser.add_argument(
        "--use_length",
        type=int,
        default=50,
        help="Chunk length to use"
    )

    parser.add_argument(
        "--chunk_ret",
        type=str2bool,
        default=True,
        help="Return chunk actions"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=55555,
        help="WebSocket server port"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="WebSocket server host"
    )

    parser.add_argument(
        "--use_compile",
        type=str2bool,
        default=True,
        help="Use torch.compile"
    )

    parser.add_argument(
        "--use_bf16",
        type=str2bool,
        default=True,
        help="Use bfloat16 precision"
    )

    parser.add_argument(
        "--use_fp32",
        type=str2bool,
        default=False,
        help="Use float32 precision"
    )

    args = parser.parse_args()

    set_seed_everywhere(42)

    model = LingbotVLAv2InferenceServer(
        args.model_path,
        base_model_path=args.base_model_path,
        robot_norm_path=args.robot_norm_path,
        use_length=args.use_length,
        chunk_ret=args.chunk_ret,
        use_bf16=args.use_bf16,
        use_fp32=args.use_fp32,
        use_compile=args.use_compile,
    )

    model_server = WebsocketInferenceServer(model, host=args.host, port=args.port)
    model_server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()