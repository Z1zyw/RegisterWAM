from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import torch

from .register_wam_processor import RegisterWAMProcessor
from register_wam.utils.logging_config import get_logger

logger = get_logger(__name__)


class RegisterWAMProcessorFast(RegisterWAMProcessor):
    def __init__(self, *args, num_image_steps: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_image_steps = None if num_image_steps is None else int(num_image_steps)

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sample = {}
        sample["instruction"] = self.augment_instruction(data)
        sample["image_is_pad"] = data["image_is_pad"]

        processed_images = []
        actual_image_steps = None
        for meta in self.shape_meta["images"]:
            key, shape = meta["key"], meta["shape"]
            image = data["images"][key]
            assert image.ndim == 4, f"Expected 4 dimensions (num_obs_steps, C, H, W), got shape {image.shape}"

            transforms = self.train_transforms if self.is_train else self.val_transforms
            current_transforms = transforms[key] if isinstance(transforms, dict) else transforms
            for trans in current_transforms:
                image = trans(image)

            if actual_image_steps is None:
                actual_image_steps = int(image.shape[0])
            elif int(image.shape[0]) != actual_image_steps:
                raise AssertionError(
                    f"All image keys must share the same temporal length, got {actual_image_steps} and {int(image.shape[0])}"
                )

            expected_image_steps = actual_image_steps if self.num_image_steps is None else self.num_image_steps
            meta_shape = [expected_image_steps] + shape
            assert list(image.shape) == meta_shape, \
                f"Expected shape {meta_shape}, got {list(image.shape)} after transforms for key {key}"

            processed_images.append(image)
        pixel_values = torch.stack(processed_images, dim=0)

        if self.num_output_cameras > pixel_values.shape[0]:
            out = torch.zeros((self.num_output_cameras,) + pixel_values.shape[1:], device=pixel_values.device, dtype=pixel_values.dtype)
            out[0: pixel_values.shape[0]] = pixel_values
            sample["pixel_values"] = out
        elif self.num_output_cameras < pixel_values.shape[0]:
            logger.warning(
                f"num_output_cameras {self.num_output_cameras} is less than the number of cameras in data {pixel_values.shape[0]}, "
                f"truncating the input to the first {self.num_output_cameras} cameras."
            )
            sample["pixel_values"] = pixel_values[:self.num_output_cameras]
        else:
            sample["pixel_values"] = pixel_values

        if not self.is_train and "action" in data:
            sample["gt_action"] = deepcopy(data["action"])

        if "action" in data and self.delta_action_dim_mask is not None:
            action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
            if bool(action_is_pad.any().item()):
                for key, dim_mask in self.delta_action_dim_mask.items():
                    cur_action = data["action"][key]
                    cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                    cur_dim_mask = dim_mask.to(device=cur_action.device)
                    pad_delta_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
                    cur_action[pad_delta_mask] = 0.0
        data = self.action_state_transform(data)
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        if "action" in data:
            sample["action"] = data["action"]
            sample["action_is_pad"] = data["action_is_pad"]
            sample["action_dim_is_pad"] = data["action_dim_is_pad"]
            assert sample["action"].shape[-1] == self.action_output_dim

        sample["proprio"] = data["state"]
        sample["proprio_is_pad"] = data["state_is_pad"]
        sample["proprio_dim_is_pad"] = data["state_dim_is_pad"]
        assert sample["proprio"].shape[-1] == self.proprio_output_dim

        sample["idx"] = data["idx"]
        return sample
