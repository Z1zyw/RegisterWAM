"""Exercise real DiT blocks without downloading weights or running a simulator."""
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from register_wam.models.wan22.action_dit import ActionDiT
from register_wam.models.wan22.mot import MoT
from register_wam.models.wan22.register_wam import RegisterWAM
from register_wam.models.wan22.wan_video_dit import WanVideoDiT
from register_wam.trainer import Wan22Trainer


class TinyVAE(torch.nn.Module):
    temporal_downsample_factor = 4

    def encode(self, video, **kwargs):
        if isinstance(video, list):
            return [self.encode(item.unsqueeze(0))[0] for item in video]
        return video[:, :2, ::4, ::8, ::8]


def test_training_checkpoint_and_cached_inference():
    torch.manual_seed(42)
    video = WanVideoDiT(hidden_dim=32, in_dim=2, ffn_dim=64, out_dim=2,
                        text_dim=16, freq_dim=8, eps=1e-6, patch_size=(1, 2, 2),
                        num_heads=2, attn_head_dim=24, num_layers=2,
                        has_image_input=False, seperated_timestep=True,
                        fuse_vae_embedding_in_latents=True)
    action = ActionDiT(hidden_dim=16, action_dim=7, ffn_dim=32, text_dim=16,
                       freq_dim=8, eps=1e-6, num_heads=2, attn_head_dim=24, num_layers=2)
    model = RegisterWAM(video, action, MoT({"video": video, "action": action},
                         mot_checkpoint_mixed_attn=False), TinyVAE(), text_dim=16,
                         proprio_dim=8, register_num_tokens=3, input_video_size=(32, 32))
    sample = {"video": torch.randn(1, 3, 9, 32, 32), "action": torch.randn(1, 32, 7),
              "context": torch.randn(1, 4, 16), "context_mask": torch.ones(1, 4, dtype=torch.bool),
              "proprio": torch.randn(1, 1, 8)}
    optimizer = torch.optim.AdamW(model.mot.parameters(), lr=1e-3)
    loss, _ = model.training_loss(sample)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.mot.register_tokens.grad is not None
    assert model.mot.register_tokens.grad.abs().sum() > 0
    optimizer.step()
    model.eval()
    inputs = model.build_inputs(sample)
    video_pre = video.pre_dit(x=inputs["first_frame_latents"], timestep=torch.zeros(1),
        context=inputs["context"], context_mask=inputs["context_mask"], action=None,
        fuse_vae_embedding_in_latents=True)
    action_pre = action.pre_dit(action_tokens=sample["action"], timestep=torch.ones(1),
        context=inputs["context"], context_mask=inputs["context_mask"])
    full, _ = model._run_mot(video_pre, action_pre)
    visual, obs_length = model._insert_registers(video_pre)
    length = visual["tokens"].shape[1]
    mask = model._register_attention_mask(length, 32, obs_length, torch.device("cpu"))
    cache = model.mot.prefill_video_cache(visual["tokens"], visual["freqs"], visual["t_mod"],
        {"context": visual["context"], "mask": visual["context_mask"]}, mask[:length, :length])
    cached = model.mot.forward_action_with_video_cache(action_pre["tokens"], action_pre["freqs"],
        action_pre["t_mod"], {"context": action_pre["context"], "mask": action_pre["context_mask"]},
        cache, mask, length)
    torch.testing.assert_close(full["action"], cached)
    kwargs = dict(prompt=None, input_image=sample["video"][:, :, 0], action_horizon=32,
                  context=sample["context"][0], context_mask=sample["context_mask"][0],
                  proprio=sample["proprio"][0, 0], num_inference_steps=2, seed=42)
    prediction = model.infer_action(**kwargs)["action"]
    assert prediction.shape == (32, 7) and torch.isfinite(prediction).all()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "weights.pt"
        model.save_checkpoint(path)
        model.load_checkpoint(path)
        torch.testing.assert_close(prediction, model.infer_action(**kwargs)["action"])
        class Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return {key: value[0] for key, value in sample.items()}

        cfg = OmegaConf.create({"output_dir": directory, "learning_rate": 1e-3,
            "weight_decay": 0.01, "batch_size": 1, "num_workers": 0, "num_epochs": 1,
            "max_steps": 2, "log_every": 1, "save_every": 1, "eval_every": None,
            "eval_num_inference_steps": 2, "gradient_accumulation_steps": 1,
            "max_grad_norm": 1, "seed": 42, "resume": None, "mixed_precision": "no",
            "lr_scheduler_type": "constant", "warmup_ratio": 0,
            "checkpoint_state_final_only": True, "checkpoint_keep_last_weights": 3,
            "wandb": {"enabled": False}})
        from accelerate import Accelerator
        with patch("register_wam.trainer.Accelerator", partial(Accelerator, cpu=True)):
            trainer = Wan22Trainer(model, Dataset(), cfg=cfg)
        trainer.train()
        assert trainer.global_step == 2
        assert (Path(directory) / "checkpoints/weights/step_000002.pt").is_file()


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_training_checkpoint_and_cached_inference()
    print("training/backward/checkpoint/cached inference: passed")
