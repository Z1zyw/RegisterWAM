"""Public model factory for clean RegisterWAM."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf

from .models.wan22.register_wam import RegisterWAM


def _as_dict(value, name: str, required: bool = False) -> dict:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        if required:
            raise ValueError(f"{name} is required.")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must resolve to a dict, got {type(value)}.")
    return value


def create_register_wam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    action_dit_config,
    register_num_tokens: int = 8,
    input_video_size: tuple[int, int] = (224, 448),
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    noised_obs=None,
    future_noise=None,
    visual_attention_mode: str = "bidirectional",
    mot_checkpoint_mixed_attn: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Build independent experts with an optional isolated observation auxiliary loss."""
    video_dit_config = _as_dict(video_dit_config, "video_dit_config", required=True)
    action_dit_config = _as_dict(action_dit_config, "action_dit_config", required=True)
    video_scheduler = _as_dict(video_scheduler, "video_scheduler")
    action_scheduler = _as_dict(action_scheduler, "action_scheduler", required=True)
    loss = _as_dict(loss, "loss")
    required = ("train_shift", "infer_shift", "num_train_timesteps")
    missing = [key for key in required if key not in action_scheduler]
    if missing:
        raise ValueError(f"action_scheduler is missing keys: {missing}")

    model = RegisterWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 0.2)),
        register_num_tokens=int(register_num_tokens),
        input_video_size=tuple(input_video_size),
    )
    noised_obs = _as_dict(noised_obs, "noised_obs")
    future_noise = _as_dict(future_noise, "future_noise")
    model.noised_obs_enabled = bool(noised_obs.get("enabled", True))
    model.noised_obs_timestep = float(noised_obs.get("timestep", 500))
    model.loss_lambda_noised_obs = float(loss.get("lambda_noised_obs", 2.0))
    model.include_first_obs_video_loss = bool(loss.get("include_first_obs_video_loss", True))
    model.future_noise_mode = str(future_noise.get("train_mode", "pure_noise"))
    model.future_loss_weight = float(future_noise.get("loss_weight", 2.0))
    if model.future_noise_mode not in {"sample", "pure_noise"}:
        raise ValueError("future_noise.train_mode must be sample or pure_noise")
    # Existing training runs use the former experiment name.
    if visual_attention_mode == "std3":
        visual_attention_mode = "bidirectional"
    if visual_attention_mode not in {"bidirectional", "obs_register_only"}:
        raise ValueError("Unsupported visual_attention_mode")
    model.visual_attention_mode = visual_attention_mode
    return model
