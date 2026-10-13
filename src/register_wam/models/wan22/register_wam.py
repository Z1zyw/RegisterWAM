from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from register_wam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class RegisterWAM(torch.nn.Module):
    """RegisterWAM with independent Wan2.2 video and action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        register_num_tokens: int = 8,
        input_video_size: tuple[int, int] = (224, 448),
    ):
        super().__init__()
        self.mot = mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.input_video_size = tuple(int(value) for value in input_video_size)
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.noised_obs_enabled = False
        self.noised_obs_timestep = 500.0
        self.loss_lambda_noised_obs = 2.0
        self.include_first_obs_video_loss = False
        self.future_noise_mode = "sample"
        self.future_loss_weight = 2.0
        self.visual_attention_mode = "obs_register_only"

        if register_num_tokens <= 0:
            raise ValueError("`register_num_tokens` must be positive.")
        self.register_num_tokens = int(register_num_tokens)
        # Match legacy CPU FP32 initialization before converting to model precision.
        self.mot.register_tokens = nn.Parameter(
            (torch.randn(1, self.register_num_tokens, int(self.video_expert.hidden_dim),
                         device="cpu", dtype=torch.float32)
             / (int(self.video_expert.hidden_dim) ** 0.5)).to(device=self.device, dtype=self.torch_dtype)
        )

        self.to(self.device)

    @property
    def video_expert(self):
        return self.mot.mixtures["video"]

    @property
    def action_expert(self):
        return self.mot.mixtures["action"]

    @property
    def dit(self):
        return self.mot

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        register_num_tokens: int = 8,
        input_video_size: tuple[int, int] = (224, 448),
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for RegisterWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for RegisterWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            register_num_tokens=register_num_tokens,
            input_video_size=input_video_size,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def _resize_center_crop_batched(self, frames):
        target_h, target_w = self.input_video_size
        in_h, in_w = frames.shape[-2:]
        scale = max(target_h / in_h, target_w / in_w)
        resized_h = max(target_h, int(in_h * scale + 0.5))
        resized_w = max(target_w, int(in_w * scale + 0.5))
        if (resized_h, resized_w) != (in_h, in_w):
            frames = F.interpolate(frames, size=(resized_h, resized_w), mode="bilinear", align_corners=False, antialias=True)
        top = max((resized_h - target_h) // 2, 0)
        left = max((resized_w - target_w) // 2, 0)
        return frames[:, :, top:top + target_h, left:left + target_w]

    def _resize_horizontal_multicam_batched(self, frames):
        target_h, target_w = self.input_video_size
        if target_w % 2 or frames.shape[-1] % 2:
            return self._resize_center_crop_batched(frames)
        half = frames.shape[-1] // 2
        cameras = (frames[:, :, :, :half], frames[:, :, :, half:])
        return torch.cat([
            F.interpolate(camera, size=(target_h, target_w // 2), mode="bilinear", align_corners=False, antialias=True)
            for camera in cameras
        ], dim=-1)

    def preprocess_video(self, video):
        """Resize raw composed camera frames on device and normalize to [-1, 1]."""
        added_batch = video.ndim == 4
        if added_batch:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"`video` must be 4D or 5D, got shape {tuple(video.shape)}")
        video = video.to(self.device, non_blocking=True)
        normalized = (video.dtype != torch.uint8 and video.numel() > 0
                      and float(video.min().item()) >= -1.01
                      and float(video.max().item()) <= 1.01
                      and float(video.min().item()) < -1e-4)
        video = video.float() / 255.0 if video.dtype == torch.uint8 else video.float()
        batch, channels, time, height, width = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        if width == 2 * height and self.input_video_size[1] == 2 * self.input_video_size[0]:
            frames = self._resize_horizontal_multicam_batched(frames)
        else:
            frames = self._resize_center_crop_batched(frames)
        if not normalized:
            frames = frames * 2.0 - 1.0
        result = frames.reshape(batch, time, channels, *self.input_video_size).permute(0, 2, 1, 3, 4).contiguous().to(self.torch_dtype)
        return result.squeeze(0) if added_batch else result

    def build_inputs(self, sample, tiled: bool = False):
        video = self.preprocess_video(sample["video"])
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "RegisterWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for RegisterWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }


    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum


    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            mot_state = payload["mot"]
            incompatible = self.mot.load_state_dict(mot_state, strict=True)
            missing_keys = list(getattr(incompatible, "missing_keys", []))
            unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
            if missing_keys or unexpected_keys:
                logger.warning(
                    "Loaded `mot` checkpoint with missing_keys=%d unexpected_keys=%d. "
                    "First missing=%s first unexpected=%s",
                    len(missing_keys),
                    len(unexpected_keys),
                    missing_keys[:20],
                    unexpected_keys[:20],
                )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    @staticmethod
    def _expand_t_mod(t_mod: torch.Tensor, sequence_length: int) -> torch.Tensor:
        if t_mod.ndim == 3:
            return t_mod.unsqueeze(1).expand(-1, sequence_length, -1, -1).contiguous()
        if t_mod.ndim == 4 and t_mod.shape[1] == sequence_length:
            return t_mod
        raise ValueError(f"Unexpected time-modulation shape {tuple(t_mod.shape)} for sequence={sequence_length}.")

    def _insert_registers(self, video_pre: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Insert registers after the first latent observation frame."""
        tokens = video_pre["tokens"]
        batch_size, video_length, _ = tokens.shape
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        obs_length = min(tokens_per_frame, video_length)
        registers = self.mot.register_tokens.to(device=tokens.device, dtype=tokens.dtype).expand(batch_size, -1, -1)

        augmented = dict(video_pre)
        augmented["tokens"] = torch.cat((tokens[:, :obs_length], registers, tokens[:, obs_length:]), dim=1)
        context_mask = video_pre["context_mask"]
        if context_mask is not None and context_mask.ndim >= 3 and context_mask.shape[-2] == video_length:
            prefix = context_mask[..., :obs_length, :]
            register_mask = context_mask[..., :1, :].expand(*context_mask.shape[:-2], self.register_num_tokens, context_mask.shape[-1])
            augmented["context_mask"] = torch.cat((prefix, register_mask, context_mask[..., obs_length:, :]), dim=-2)
        register_freqs = video_pre["freqs"][:1].expand(self.register_num_tokens, -1, -1).contiguous()
        augmented["freqs"] = torch.cat(
            (video_pre["freqs"][:obs_length], register_freqs, video_pre["freqs"][obs_length:]), dim=0
        )
        expanded_t_mod = self._expand_t_mod(video_pre["t_mod"], video_length)
        register_t_mod = expanded_t_mod[:, :1].expand(-1, self.register_num_tokens, -1, -1).contiguous()
        augmented["t_mod"] = torch.cat(
            (expanded_t_mod[:, :obs_length], register_t_mod, expanded_t_mod[:, obs_length:]), dim=1
        )
        return augmented, obs_length

    def _remove_registers(self, tokens: torch.Tensor, obs_length: int) -> torch.Tensor:
        return torch.cat(
            (tokens[:, :obs_length], tokens[:, obs_length + self.register_num_tokens:]), dim=1
        )

    def _register_attention_mask(
        self,
        augmented_video_length: int,
        action_length: int,
        obs_length: int,
        device: torch.device,
        noised_obs_length: int = 0,
    ) -> torch.Tensor:
        """Build the only cross-expert contract used by this implementation.

        The bidirectional mode connects observation, register, and future
        tokens. Auxiliary observations remain isolated. Action tokens see
        only action tokens and registers; visual tokens never see actions.
        """
        register_start = obs_length
        register_end = register_start + self.register_num_tokens
        total = augmented_video_length + action_length
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        # Observation/register workspace: no access to future video tokens.
        mask[:register_end, :register_end] = True
        # Future video may read its observation/register context and other future tokens.
        mask[register_end:augmented_video_length, :augmented_video_length] = True
        main_length = augmented_video_length - noised_obs_length
        if self.visual_attention_mode in {"bidirectional", "std3"}:
            mask[:main_length, :main_length] = True
        if noised_obs_length:
            mask[:augmented_video_length, main_length:augmented_video_length] = False
            mask[main_length:augmented_video_length, :augmented_video_length] = False
            mask[main_length:augmented_video_length, main_length:augmented_video_length] = True
        # Action is autoregressively independent in token space and receives the registers.
        mask[augmented_video_length:, augmented_video_length:] = True
        mask[augmented_video_length:, register_start:register_end] = True
        return mask

    def _append_noised_obs(self, visual_pre, noised_pre):
        if noised_pre is None:
            return visual_pre
        combined = dict(visual_pre)
        for key in ("tokens", "t_mod"):
            extra = noised_pre[key]
            if key == "t_mod":
                extra = self._expand_t_mod(extra, noised_pre["tokens"].shape[1])
            combined[key] = torch.cat((visual_pre[key], extra), dim=1)
        combined["freqs"] = torch.cat((visual_pre["freqs"], noised_pre["freqs"]), dim=0)
        mask = visual_pre["context_mask"]
        if mask is not None and mask.ndim >= 3 and mask.shape[-2] != 1:
            combined["context_mask"] = torch.cat((mask, noised_pre["context_mask"]), dim=-2)
        return combined

    def _run_mot(self, video_pre: dict[str, Any], action_pre: dict[str, Any], noised_pre=None) -> tuple[dict[str, torch.Tensor], int]:
        augmented_video_pre, obs_length = self._insert_registers(video_pre)
        main_length = augmented_video_pre["tokens"].shape[1]
        augmented_video_pre = self._append_noised_obs(augmented_video_pre, noised_pre)
        attention_mask = self._register_attention_mask(
            augmented_video_length=int(augmented_video_pre["tokens"].shape[1]),
            action_length=int(action_pre["tokens"].shape[1]),
            obs_length=obs_length,
            device=augmented_video_pre["tokens"].device,
            noised_obs_length=0 if noised_pre is None else noised_pre["tokens"].shape[1],
        )
        visual_length = augmented_video_pre["tokens"].shape[1]
        # Visual never reads action; retain cache gradients for the action loss.
        cache, visual_output = self.mot.prefill_video_cache(
            augmented_video_pre["tokens"], augmented_video_pre["freqs"], augmented_video_pre["t_mod"],
            {"context": augmented_video_pre["context"], "mask": augmented_video_pre["context_mask"]},
            attention_mask[:visual_length, :visual_length], return_tokens=True,
        )
        action_output = self.mot.forward_action_with_video_cache(
            action_pre["tokens"], action_pre["freqs"], action_pre["t_mod"],
            {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            cache, attention_mask, visual_length,
            video_kv_slice=(obs_length, obs_length + self.register_num_tokens),
        )
        output = {"video": visual_output, "action": action_output}
        if noised_pre is not None:
            output["noised_obs"] = output["video"][:, main_length:]
        output["video"] = self._remove_registers(output["video"][:, :main_length], obs_length)
        return output, obs_length

    def training_loss(self, sample: dict[str, Any], tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        action = inputs["action"]

        noise_video = torch.randn_like(input_latents)
        if self.future_noise_mode == "pure_noise":
            timestep_video = torch.full(
                (batch_size,), float(self.train_video_scheduler.num_train_timesteps),
                device=self.device, dtype=input_latents.dtype,
            )
        else:
            timestep_video = self.train_video_scheduler.sample_training_t(batch_size, self.device, input_latents.dtype)
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if self.include_first_obs_video_loss:
            target_video[:, :, :1] = -input_latents[:, :, :1]
        if inputs["first_frame_latents"] is not None:
            latents[:, :, :1] = inputs["first_frame_latents"]

        # Match the legacy RNG order: video, noised observation, then action.
        if self.noised_obs_enabled:
            obs = input_latents[:, :, :1].contiguous()
            obs_noise = torch.randn_like(obs)
            obs_t = torch.full((batch_size,), self.noised_obs_timestep, device=self.device, dtype=obs.dtype)
            noised_obs_latents = self.train_video_scheduler.add_noise(obs, obs_noise, obs_t)
            target_obs = self.train_video_scheduler.training_target(obs, obs_noise, obs_t)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(batch_size, self.device, action.dtype)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents, timestep=timestep_video, context=inputs["context"], context_mask=inputs["context_mask"],
            action=action, fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action, timestep=timestep_action,
            context=inputs["context"], context_mask=inputs["context_mask"],
        )
        noised_pre = None
        if self.noised_obs_enabled:
            noised_pre = self.video_expert.pre_dit(
                x=noised_obs_latents, timestep=obs_t,
                context=inputs["context"], context_mask=inputs["context_mask"], action=action,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )
        tokens_out, _ = self._run_mot(video_pre, action_pre, noised_pre)
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = self.include_first_obs_video_loss or inputs["first_frame_latents"] is None
        if not include_initial_video_step:
            pred_video, target_video = pred_video[:, :, 1:], target_video[:, :, 1:]
        video_per_sample = self._compute_video_loss_per_sample(
            pred_video, target_video, inputs["image_is_pad"], include_initial_video_step
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(video_per_sample)
        if self.future_noise_mode == "pure_noise":
            video_weight = torch.full_like(video_per_sample, self.future_loss_weight)
        loss_video = (video_per_sample * video_weight).mean()

        action_per_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        action_is_pad = inputs["action_is_pad"]
        if action_is_pad is None:
            action_per_sample = action_per_token.mean(dim=1)
        else:
            valid = (~action_is_pad).to(action_per_token)
            action_per_sample = (action_per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(action_per_sample)
        loss_action = (action_per_sample * action_weight).mean()

        loss_obs = loss_video.new_zeros(())
        if noised_pre is not None:
            pred_obs = self.video_expert.post_dit(tokens_out["noised_obs"], noised_pre)
            obs_error = F.mse_loss(pred_obs.float(), target_obs.float(), reduction="none").mean(dim=(1, 2, 3, 4))
            loss_obs = (obs_error * self.train_video_scheduler.training_weight(obs_t).to(obs_error)).mean()
        total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action + self.loss_lambda_noised_obs * loss_obs
        return total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach()),
            "loss_noised_obs": self.loss_lambda_noised_obs * float(loss_obs.detach()),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        num_video_frames: int = 9,
        **unused: Any,
    ) -> dict[str, Any]:
        del unused
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[:2] != (1, 3):
            raise ValueError(f"`input_image` must be [1,3,H,W], got {tuple(input_image.shape)}")
        input_image = self.preprocess_video(input_image.unsqueeze(2))[:, :, 0]
        if context is None or context_mask is None:
            if prompt is None:
                raise ValueError("Provide either a prompt or precomputed context/context_mask.")
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio.to(self.device))

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_latents = torch.randn(
            (1, action_horizon, self.action_expert.action_dim), generator=generator,
            device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        obs_latents = self._encode_input_image_latents_tensor(input_image.to(self.device, self.torch_dtype), tiled=tiled)
        video_timestep = torch.zeros((1,), device=self.device, dtype=obs_latents.dtype)
        visual_latents = obs_latents
        if self.visual_attention_mode in {"bidirectional", "std3"}:
            if num_video_frames < 1 or num_video_frames % 4 != 1:
                raise ValueError("num_video_frames must satisfy T % 4 == 1")
            latent_frames = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
            visual_latents = torch.randn(
                (1, obs_latents.shape[1], latent_frames, *obs_latents.shape[3:]),
                generator=None if seed is None else torch.Generator(device=rand_device).manual_seed(seed),
                device=rand_device, dtype=torch.float32,
            ).to(obs_latents)
            visual_latents[:, :, :1] = obs_latents
            video_timestep.fill_(self.infer_video_scheduler.num_train_timesteps)
        video_pre = self.video_expert.pre_dit(
            x=visual_latents, timestep=video_timestep, context=context, context_mask=context_mask, action=None,
            fuse_vae_embedding_in_latents=bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)),
        )
        visual_pre, obs_length = self._insert_registers(video_pre)
        noised_pre = None
        if self.noised_obs_enabled:
            obs_noise = torch.randn(obs_latents.shape,
                generator=None if seed is None else torch.Generator(device=rand_device).manual_seed(seed + 991),
                device=rand_device, dtype=torch.float32).to(obs_latents)
            obs_t = torch.full_like(video_timestep, self.noised_obs_timestep)
            noised_pre = self.video_expert.pre_dit(
                x=self.infer_video_scheduler.add_noise(obs_latents, obs_noise, obs_t), timestep=obs_t,
                context=context, context_mask=context_mask, action=None,
                fuse_vae_embedding_in_latents=bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)),
            )
            visual_pre = self._append_noised_obs(visual_pre, noised_pre)
        visual_length = visual_pre["tokens"].shape[1]
        attention_mask = self._register_attention_mask(
            visual_length, action_horizon, obs_length, self.device,
            noised_obs_length=0 if noised_pre is None else noised_pre["tokens"].shape[1],
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=visual_pre["tokens"],
            video_freqs=visual_pre["freqs"],
            video_t_mod=visual_pre["t_mod"],
            video_context_payload={"context": visual_pre["context"], "mask": visual_pre["context_mask"]},
            video_attention_mask=attention_mask[:visual_length, :visual_length],
        )
        schedule, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=self.device, dtype=action_latents.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(schedule, deltas):
            action_pre = self.action_expert.pre_dit(
                action_tokens=action_latents, timestep=timestep.unsqueeze(0), context=context, context_mask=context_mask,
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=visual_length,
                video_kv_slice=(obs_length, obs_length + self.register_num_tokens),
            )
            prediction = self.action_expert.post_dit(action_tokens, action_pre)
            action_latents = self.infer_action_scheduler.step(prediction, delta, action_latents)
        return {"action": action_latents[0].float().cpu()}

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
