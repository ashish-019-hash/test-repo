import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoProcessor

from gr00t.data.types import (
    ModalityConfig,
    ActionConfig,
    ActionRepresentation,
    ActionType,
    ActionFormat,
    EmbodimentTag,
    MessageType,
    VLAStepData,
)
from gr00t.data.interfaces import BaseProcessor
from gr00t.policy.server_client import PolicyServer
from gr00t.policy.policy import BasePolicy

SPOT_MODALITY_CONFIGS = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front_camera"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_positions", "joint_velocities"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["base_velocity"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

SPOT_DUMMY_STATISTICS = {
    EmbodimentTag.NEW_EMBODIMENT.value: {
        "state": {
            "joint_positions": {
                "min": [-3.14] * 12,
                "max": [3.14] * 12,
                "mean": [0.0] * 12,
                "std": [1.0] * 12,
                "q01": [-3.14] * 12,
                "q99": [3.14] * 12,
            },
            "joint_velocities": {
                "min": [-10.0] * 12,
                "max": [10.0] * 12,
                "mean": [0.0] * 12,
                "std": [2.0] * 12,
                "q01": [-10.0] * 12,
                "q99": [10.0] * 12,
            },
        },
        "action": {
            "base_velocity": {
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0],
                "mean": [0.0, 0.0, 0.0],
                "std": [0.3, 0.3, 0.3],
                "q01": [-1.0, -1.0, -1.0],
                "q99": [1.0, 1.0, 1.0],
            },
        },
    }
}


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x



PERCEPTION_SCALE = 0.5


class SpotGr00tPolicy(BasePolicy):
    def __init__(self, model_path, device="cuda", novelty_threshold=0.15):
        super().__init__(strict=False)
        import gr00t.model  # noqa: F401

        model_dir = Path(model_path)
        print("Loading model...")
        self.model = AutoModel.from_pretrained(model_dir)
        self.model.eval()
        self.model.to(device=device, dtype=torch.bfloat16)

        print("Loading processor...")
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(model_dir)
        self.processor.eval()

        tag = EmbodimentTag.NEW_EMBODIMENT.value

        self.processor.modality_configs[tag] = SPOT_MODALITY_CONFIGS
        self.processor.state_action_processor.modality_configs[tag] = SPOT_MODALITY_CONFIGS

        if tag not in self.processor.embodiment_id_mapping:
            self.processor.embodiment_id_mapping[tag] = 10

        self.processor.set_statistics(SPOT_DUMMY_STATISTICS)
        print(f"Injected Spot config for '{tag}'.")

        self.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
        self.modality_configs = SPOT_MODALITY_CONFIGS
        self.collate_fn = self.processor.collator
        self.device = device

        self._novelty_threshold = novelty_threshold
        self._prev_feature = None
        self._reference_feature = None
        self._warmup_features = []
        self._feature_query_count = 0
        self._feature_warmup_count = 3

    def _build_fallback_action(self):
        action_horizon = len(self.modality_configs["action"].delta_indices)
        return {"action.base_velocity": np.zeros((action_horizon, 3), dtype=np.float32)}

    def check_observation(self, observation):
        pass

    def check_action(self, action):
        pass

    def _downscale_image(self, image):
        h, w = image.shape[:2]
        new_h, new_w = int(h * PERCEPTION_SCALE), int(w * PERCEPTION_SCALE)
        img_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        scaled = F.interpolate(img_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        canvas = torch.zeros(1, image.shape[2], h, w)
        y_off = (h - new_h) // 2
        x_off = (w - new_w) // 2
        canvas[:, :, y_off:y_off + new_h, x_off:x_off + new_w] = scaled
        return canvas.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

    def _get_action(self, observation, options=None):
        try:
            images = {}
            states = {}
            text = None

            for key, value in observation.items():
                if key.startswith("video."):
                    k = key[len("video."):]
                    if isinstance(value, np.ndarray):
                        if value.ndim == 3:
                            images[k] = [self._downscale_image(value)]
                        elif value.ndim == 4:
                            images[k] = [self._downscale_image(value[i]) for i in range(value.shape[0])]
                elif key.startswith("state."):
                    k = key[len("state."):]
                    if isinstance(value, np.ndarray):
                        if value.ndim == 1:
                            states[k] = value[np.newaxis]
                        else:
                            states[k] = value
                elif "annotation" in key or "language" in key:
                    text = str(value)

            step_data = VLAStepData(
                images=images,
                states=states,
                actions={},
                text=text,
                embodiment=self.embodiment_tag,
            )

            messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
            processed = self.processor(messages)
            collated = self.collate_fn([processed])

            with torch.inference_mode():
                inputs_dict = collated["inputs"]
                backbone_inputs, action_inputs = self.model.prepare_input(inputs_dict)
                backbone_outputs = self.model.backbone(backbone_inputs)

                image_mask = backbone_outputs["image_mask"][0]
                all_features = backbone_outputs["backbone_features"][0]
                img_features = all_features[image_mask].float()
                avg_feature = F.normalize(img_features.mean(dim=0), dim=0)

                frame_novelty = 0.0
                ref_novelty = 0.0
                object_detected = False

                self._feature_query_count += 1

                if self._prev_feature is not None:
                    frame_novelty = 1.0 - F.cosine_similarity(
                        avg_feature.unsqueeze(0), self._prev_feature.unsqueeze(0)
                    ).item()

                if self._feature_query_count <= self._feature_warmup_count:
                    self._warmup_features.append(avg_feature.clone())
                    print(f"[GR00T] Warmup query {self._feature_query_count}/{self._feature_warmup_count}")
                    if self._feature_query_count == self._feature_warmup_count:
                        self._reference_feature = torch.stack(self._warmup_features).mean(dim=0)
                        self._reference_feature = F.normalize(self._reference_feature, dim=0)
                        print(f"[GR00T] Reference feature established from {self._feature_warmup_count} warmup frames")

                if self._reference_feature is not None:
                    ref_novelty = 1.0 - F.cosine_similarity(
                        avg_feature.unsqueeze(0), self._reference_feature.unsqueeze(0)
                    ).item()
                    if self._feature_query_count > self._feature_warmup_count and ref_novelty > self._novelty_threshold:
                        object_detected = True

                self._prev_feature = avg_feature.clone()

                print(f"[GR00T] Query #{self._feature_query_count} | frame_novelty={frame_novelty:.4f}, ref_novelty={ref_novelty:.4f}, threshold={self._novelty_threshold}, detected={object_detected}")

                action_outputs = self.model.action_head.get_action(backbone_outputs, action_inputs)

            normalized_action = action_outputs["action_pred"].float()

            batched_states = {}
            for k in self.modality_configs["state"].modality_keys:
                batched_states[k] = states[k][np.newaxis]

            unnormalized_action = self.processor.decode_action(
                normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
            )

            action = {f"action.{k}": v.astype(np.float32) for k, v in unnormalized_action.items()}

            return action, {
                "status": "ok",
                "object_detected": object_detected,
                "frame_novelty": round(frame_novelty, 4),
                "ref_novelty": round(ref_novelty, 4),
            }

        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return self._build_fallback_action(), {"status": "fallback", "error": str(e)}

    def get_modality_config(self):
        return self.modality_configs

    def reset(self, options=None):
        self._prev_feature = None
        self._reference_feature = None
        self._warmup_features = []
        self._feature_query_count = 0
        return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="Spot GR00T Policy Server")
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--novelty-threshold", type=float, default=0.08,
                        help="Cosine novelty threshold for object detection (0.0-1.0). Lower = more sensitive.")
    args = parser.parse_args()

    print("Starting Spot GR00T Policy Server...")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Host: {args.host}:{args.port}")
    print(f"  Novelty threshold: {args.novelty_threshold}")

    policy = SpotGr00tPolicy(
        model_path=args.model_path,
        device=args.device,
        novelty_threshold=args.novelty_threshold,
    )
    print("Model loaded successfully.")

    server = PolicyServer(policy=policy, host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main()
