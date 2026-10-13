import types
import unittest

import torch
import torch.nn.functional as F

from register_wam.models.wan22.register_wam import RegisterWAM
from test_training_smoke import TinyVAE
from register_wam.models.wan22.action_dit import ActionDiT
from register_wam.models.wan22.mot import MoT
from register_wam.models.wan22.wan_video_dit import WanVideoDiT


class TwoStreamAlignmentTest(unittest.TestCase):
    def test_preprocess_resizes_cameras_independently(self):
        stub = types.SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.float32,
                                     input_video_size=(16, 32))
        for name in ("_resize_center_crop_batched", "_resize_horizontal_multicam_batched"):
            setattr(stub, name, types.MethodType(getattr(RegisterWAM, name), stub))
        raw = torch.zeros(1, 3, 5, 40, 80, dtype=torch.uint8)
        raw[..., 40:] = 255
        output = RegisterWAM.preprocess_video(stub, raw)
        self.assertTrue((output[..., :16] == -1).all())
        torch.testing.assert_close(output[..., 16:], torch.ones_like(output[..., 16:]))
        torch.testing.assert_close(RegisterWAM.preprocess_video(stub, output), output)
        raw = torch.randint(0, 256, raw.shape, dtype=torch.uint8)
        frames = raw.float().div(255).permute(0, 2, 1, 3, 4).reshape(5, 3, 40, 80)
        expected = torch.cat([
            F.interpolate(camera, size=(16, 16), mode="bilinear", align_corners=False, antialias=True)
            for camera in frames.chunk(2, dim=-1)
        ], dim=-1).mul(2).sub(1).reshape(1, 5, 3, 16, 32).permute(0, 2, 1, 3, 4)
        torch.testing.assert_close(RegisterWAM.preprocess_video(stub, raw), expected, rtol=0, atol=0)
        torch.testing.assert_close(RegisterWAM.preprocess_video(stub, raw[0]), expected[0], rtol=0, atol=0)

    def test_training_and_inference_use_only_register_kv(self):
        torch.set_num_threads(2)
        video = WanVideoDiT(hidden_dim=32, in_dim=2, ffn_dim=64, out_dim=2,
            text_dim=16, freq_dim=8, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
            attn_head_dim=24, num_layers=2, has_image_input=False,
            seperated_timestep=True, fuse_vae_embedding_in_latents=True)
        action = ActionDiT(hidden_dim=16, action_dim=7, ffn_dim=32, text_dim=16,
            freq_dim=8, eps=1e-6, num_heads=2, attn_head_dim=24, num_layers=2)
        model = RegisterWAM(video, action, MoT({"video": video, "action": action}),
            TinyVAE(), text_dim=16, register_num_tokens=3, input_video_size=(32, 32))
        model.visual_attention_mode = "bidirectional"
        model.future_noise_mode = "pure_noise"
        model.noised_obs_enabled = True
        sample = {"video": torch.randint(0, 256, (1, 3, 9, 32, 32), dtype=torch.uint8),
                  "action": torch.randn(1, 32, 7), "context": torch.randn(1, 4, 16),
                  "context_mask": torch.ones(1, 4, dtype=torch.bool)}
        sizes = []
        original = model.mot._mixed_attention

        def record(q_cat, k_cat, v_cat, attention_mask):
            sizes.append((q_cat.shape[1], k_cat.shape[1]))
            return original(q_cat, k_cat, v_cat, attention_mask)

        model.mot._mixed_attention = record
        loss, _ = model.training_loss(sample)
        loss.backward()
        self.assertGreater(float(model.mot.register_tokens.grad.abs().sum()), 0)
        self.assertEqual(sizes, [(19, 19)] * 2 + [(32, 35)] * 2)
        sizes.clear()
        model.infer_action(prompt=None, input_image=sample["video"][:, :, 0],
            action_horizon=32, context=sample["context"], context_mask=sample["context_mask"],
            num_video_frames=9, num_inference_steps=2, seed=42)
        self.assertEqual(sizes, [(19, 19)] * 2 + [(32, 35)] * 4)


if __name__ == "__main__":
    unittest.main()
