from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from register_wam.utils import misc
from register_wam.utils.logging_config import get_logger

from ..dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from .base_lerobot_dataset_fast import BaseLerobotDatasetFast
from .lerobot.fast_patch import apply_fast_lerobot_patches
from .robot_video_dataset import DEFAULT_PROMPT
from .utils.normalizer import load_dataset_stats_from_json, save_dataset_stats_to_json

logger = get_logger(__name__)


class RobotVideoDatasetFastRawOptimized(torch.utils.data.Dataset):
    """RegisterWAM fast-path dataset.

    Compared with RobotVideoDatasetFastRaw, this path additionally:
    - patches LeRobotDataset to avoid hf_dataset.select(...)
    - decodes only the frames that survive action_video_freq_ratio subsampling
    - keeps raw composed video and leaves resize/normalize to GPU-side model preprocess
    """

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        episode_subsample_stride: int = 1,
        episode_subsample_offset: int = 0,
        max_episodes: Optional[int] = None,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: Optional[str] = None,
        text_context_cache_size: int = 4096,
        log_retry_traceback: bool = False,
        video_augmentation: Optional[dict] = None,
    ):
        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        assert (self.num_frames - 1) % self.action_video_freq_ratio == 0, (
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {self.num_frames - 1} and {self.action_video_freq_ratio}"
        )
        assert ((self.num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, (
            f"video frames must be divisible by 4 for tokenization, got {(self.num_frames - 1) // self.action_video_freq_ratio}"
        )
        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))
        self._video_sample_indices_tensor = torch.tensor(self.video_sample_indices, dtype=torch.long)

        self.lerobot_dataset = BaseLerobotDatasetFast(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=self.num_frames,
            action_size=self.num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            episode_subsample_stride=episode_subsample_stride,
            episode_subsample_offset=episode_subsample_offset,
            max_episodes=max_episodes,
            log_retry_traceback=log_retry_traceback,
        )
        apply_fast_lerobot_patches(self.lerobot_dataset.multi_dataset, video_sample_indices=self.video_sample_indices)

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = [int(video_size[0]), int(video_size[1])]
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)
        self.skip_padding_as_possible = bool(skip_padding_as_possible)
        self.max_padding_retry = int(max_padding_retry)
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.text_context_cache_size = int(text_context_cache_size)
        self.video_augmentation = self._normalize_video_augmentation(video_augmentation)

        self.resize_transform = ResizeSmallestSideAspectPreserving(args={"img_w": self.video_size[1], "img_h": self.video_size[0]})
        self.crop_transform = CenterCrop(args={"img_w": self.video_size[1], "img_h": self.video_size[0]})
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        self._cache_dir_prepared = False
        self._text_context_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._prompt_path_cache: OrderedDict[str, str] = OrderedDict()

        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    def _trim_lru(self, cache: OrderedDict):
        while len(cache) > self.text_context_cache_size:
            cache.popitem(last=False)

    def _prepare_cache_dir(self):
        if self._cache_dir_prepared:
            return
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        os.makedirs(self.text_embedding_cache_dir, exist_ok=True)
        self._cache_dir_prepared = True

    def _cache_path_for_prompt(self, prompt: str) -> str:
        cached = self._prompt_path_cache.get(prompt)
        if cached is not None:
            self._prompt_path_cache.move_to_end(prompt)
            return cached
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.text_embedding_cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        self._prompt_path_cache[prompt] = cache_path
        self._trim_lru(self._prompt_path_cache)
        return cache_path

    def _load_cached_context(self, cache_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}")
        if context_mask.ndim != 1:
            raise ValueError(f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}")
        if context.shape[0] != self.context_len:
            raise ValueError(f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}")
        if context_mask.shape[0] != self.context_len:
            raise ValueError(f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}")
        if not bool(context_mask.all()):
            context = context.clone()
            context[~context_mask] = 0.0
        full_mask = torch.ones_like(context_mask, dtype=torch.bool)
        return context.contiguous(), full_mask.contiguous()

    def _get_cached_text_context(self, prompt: str):
        self._prepare_cache_dir()
        cached = self._text_context_cache.get(prompt)
        if cached is not None:
            self._text_context_cache.move_to_end(prompt)
            return cached

        cache_path = self._cache_path_for_prompt(prompt)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. Run scripts/precompute_text_embeds.py first."
            )
        cached_value = self._load_cached_context(cache_path)
        self._text_context_cache[prompt] = cached_value
        self._trim_lru(self._text_context_cache)
        return cached_value

    def _normalize_video_augmentation(self, cfg: Optional[dict]) -> dict:
        if cfg is None:
            cfg = {}
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg, dict):
            raise TypeError(f"video_augmentation must be a mapping or None, got {type(cfg)}")

        rotate_degrees = cfg.get("rotate_degrees", 0.0)
        if isinstance(rotate_degrees, (int, float)):
            rotate_min = -abs(float(rotate_degrees))
            rotate_max = abs(float(rotate_degrees))
        elif isinstance(rotate_degrees, (list, tuple)) and len(rotate_degrees) == 2:
            rotate_min = float(rotate_degrees[0])
            rotate_max = float(rotate_degrees[1])
        else:
            raise ValueError("video_augmentation.rotate_degrees must be a number or [min, max].")

        scale_range = cfg.get("scale_range", [1.0, 1.0])
        if isinstance(scale_range, (int, float)):
            scale_min = 1.0 / float(scale_range)
            scale_max = float(scale_range)
        elif isinstance(scale_range, (list, tuple)) and len(scale_range) == 2:
            scale_min = float(scale_range[0])
            scale_max = float(scale_range[1])
        else:
            raise ValueError("video_augmentation.scale_range must be a number or [min, max].")
        if scale_min <= 0 or scale_max <= 0:
            raise ValueError("video_augmentation.scale_range values must be positive.")
        if scale_min > scale_max:
            scale_min, scale_max = scale_max, scale_min

        camera_order = cfg.get("camera_order", None)
        if camera_order is not None:
            camera_order = [int(i) for i in camera_order]

        affine_camera_indices = cfg.get("affine_camera_indices", None)
        if affine_camera_indices is not None:
            affine_camera_indices = [int(i) for i in affine_camera_indices]

        return {
            "enabled": bool(cfg.get("enabled", False)),
            "probability": float(cfg.get("probability", 0.5)),
            "swap_probability": float(cfg.get("swap_probability", 0.5)),
            "rotate_min": rotate_min,
            "rotate_max": rotate_max,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "translate_frac": float(cfg.get("translate_frac", 0.0)),
            "fill": float(cfg.get("fill", 0.0)),
            "camera_order": camera_order,
            "affine_camera_indices": affine_camera_indices,
        }

    def _maybe_augment_multicamera_video(self, video: torch.Tensor) -> torch.Tensor:
        """Apply sample-level video augmentation before camera composition.

        Input/output shape is [num_cameras, T, C, H, W]. All frames and cameras in
        one sample share the same affine parameters so temporal alignment is kept.
        """
        cfg = self.video_augmentation
        camera_order = cfg.get("camera_order")
        if camera_order is not None:
            if sorted(camera_order) != list(range(int(video.shape[0]))):
                raise ValueError(f"Invalid video_augmentation.camera_order={camera_order} for {int(video.shape[0])} cameras.")
            order = torch.as_tensor(camera_order, device=video.device, dtype=torch.long)
            video = video.index_select(0, order)

        if not (self.lerobot_dataset.is_training_set and cfg.get("enabled", False)):
            return video
        if np.random.random() >= cfg["probability"]:
            return video

        should_swap = int(video.shape[0]) == 2 and np.random.random() < cfg["swap_probability"]

        angle = float(np.random.uniform(cfg["rotate_min"], cfg["rotate_max"]))
        scale = float(np.random.uniform(cfg["scale_min"], cfg["scale_max"]))
        translate_frac = float(cfg["translate_frac"])
        if translate_frac > 0:
            height, width = int(video.shape[-2]), int(video.shape[-1])
            translate = [
                int(round(np.random.uniform(-translate_frac, translate_frac) * width)),
                int(round(np.random.uniform(-translate_frac, translate_frac) * height)),
            ]
        else:
            translate = [0, 0]

        num_cameras, num_frames = int(video.shape[0]), int(video.shape[1])
        affine_camera_indices = cfg.get("affine_camera_indices")
        if affine_camera_indices is None:
            affine_camera_indices = list(range(num_cameras))
        if affine_camera_indices:
            if min(affine_camera_indices) < 0 or max(affine_camera_indices) >= num_cameras:
                raise ValueError(
                    f"Invalid video_augmentation.affine_camera_indices={affine_camera_indices} "
                    f"for {num_cameras} cameras."
                )

            aug_video = video.clone()
            index = torch.as_tensor(affine_camera_indices, device=video.device, dtype=torch.long)
            selected = video.index_select(0, index)
            flat = selected.reshape(len(affine_camera_indices) * num_frames, *video.shape[2:])
            aug = transforms_F.affine(
                flat,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                fill=cfg["fill"],
                center=None,
            ).reshape(len(affine_camera_indices), num_frames, *video.shape[2:])
            aug_video[index] = aug
            video = aug_video

        if should_swap:
            video = video.index_select(0, torch.tensor([1, 0], device=video.device, dtype=torch.long))
        return video.contiguous()

    def _compose_multicamera_video(self, video: torch.Tensor) -> torch.Tensor:
        num_cameras = int(video.shape[0])
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}")
            cam_top = transforms_F.resize(video[0], size=[256, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
            cam_left = transforms_F.resize(video[1], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
            cam_right = transforms_F.resize(video[2], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
            bottom = torch.cat((cam_left, cam_right), dim=-1)
            return torch.cat((cam_top, bottom), dim=-2)
        if num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                return torch.cat(tuple(video.unbind(0)), dim=-1)
            if self.concat_multi_camera == "vertical":
                return torch.cat(tuple(video.unbind(0)), dim=-2)
            raise ValueError(
                f"Invalid concat_multi_camera: {self.concat_multi_camera}. Expected one of: horizontal, vertical, robotwin."
            )
        return video.squeeze(0)

    def _finalize_video(self, video: torch.Tensor) -> torch.Tensor:
        return video.permute(1, 0, 2, 3).contiguous()

    def _subsample_video_if_needed(self, video: torch.Tensor, image_is_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_len = len(self.video_sample_indices)
        if video.ndim == 5:
            decoded_len = int(video.shape[1])
            if decoded_len == self.num_frames:
                index_tensor = self._video_sample_indices_tensor.to(video.device)
                video = video.index_select(1, index_tensor)
            elif decoded_len != target_len:
                raise ValueError(f"Unexpected decoded frame count {decoded_len}, expected either {self.num_frames} or {target_len}")
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            decoded_len = int(video.shape[0])
            if decoded_len == self.num_frames:
                index_tensor = self._video_sample_indices_tensor.to(video.device)
                video = video.index_select(0, index_tensor)
            elif decoded_len != target_len:
                raise ValueError(f"Unexpected decoded frame count {decoded_len}, expected either {self.num_frames} or {target_len}")
            video = video.unsqueeze(0)

        if int(image_is_pad.shape[0]) == self.num_frames:
            image_is_pad = image_is_pad[self.video_sample_indices]
        elif int(image_is_pad.shape[0]) != target_len:
            raise ValueError(
                f"Unexpected image_is_pad length {int(image_is_pad.shape[0])}, expected either {self.num_frames} or {target_len}"
            )
        return video, image_is_pad

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            if not self.skip_padding_as_possible:
                break
            has_pad = bool(sample["action_is_pad"].any()) or bool(sample["image_is_pad"].any()) or bool(sample["proprio_is_pad"].any())
            if not has_pad or attempt >= self.max_padding_retry:
                break
            sample_idx = int(np.random.randint(len(self.lerobot_dataset)))

        image_is_pad = sample["image_is_pad"]
        video = sample["pixel_values"]
        video, image_is_pad = self._subsample_video_if_needed(video, image_is_pad)
        video = self._maybe_augment_multicamera_video(video)

        video = self._compose_multicamera_video(video)
        video = self._finalize_video(video)

        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._get_cached_text_context(instruction)

        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
            "prompt": instruction,
            "instruction": instruction,
            "context": context,
            "context_mask": context_mask,
        }

    def __getitem__(self, idx):
        return self._get(idx)
