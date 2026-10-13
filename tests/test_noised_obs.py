import unittest

import torch

from register_wam.models.wan22.action_dit import ActionDiT
from register_wam.models.wan22.mot import MoT
from register_wam.models.wan22.register_wam import RegisterWAM
from register_wam.models.wan22.wan_video_dit import WanVideoDiT
from test_training_smoke import TinyVAE


class NoisedObsTest(unittest.TestCase):
    def test_switch_loss_mask_and_inference(self):
        torch.set_num_threads(2)
        video = WanVideoDiT(hidden_dim=32, in_dim=2, ffn_dim=64, out_dim=2,
            text_dim=16, freq_dim=8, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
            attn_head_dim=24, num_layers=2, has_image_input=False,
            seperated_timestep=True, fuse_vae_embedding_in_latents=True)
        action = ActionDiT(hidden_dim=16, action_dim=7, ffn_dim=32, text_dim=16,
            freq_dim=8, eps=1e-6, num_heads=2, attn_head_dim=24, num_layers=2)
        model = RegisterWAM(video, action, MoT({"video": video, "action": action},
            mot_checkpoint_mixed_attn=False), TinyVAE(), text_dim=16,
            register_num_tokens=3, input_video_size=(32, 32), loss_lambda_action=0.2)
        model.visual_attention_mode = "bidirectional"
        model.future_noise_mode = "pure_noise"
        model.include_first_obs_video_loss = True
        sample = {"video": torch.randn(2, 3, 9, 32, 32), "action": torch.randn(2, 32, 7),
            "context": torch.randn(2, 4, 16), "context_mask": torch.ones(2, 4, dtype=torch.bool),
            "image_is_pad": torch.zeros(2, 9, dtype=torch.bool)}
        calls = []
        handle = video.patch_embedding.register_forward_hook(lambda *args: calls.append(1))
        for enabled in (False, True):
            model.noised_obs_enabled = enabled
            calls.clear()
            model.train()
            loss, metrics = model.training_loss(sample)
            self.assertEqual(len(calls), 1 + int(enabled))
            self.assertEqual(metrics["loss_noised_obs"] > 0, enabled)
            self.assertAlmostEqual(float(loss.detach()), sum(metrics.values()), places=5)
            loss.backward()
            self.assertTrue(torch.isfinite(model.mot.register_tokens.grad).all())
            model.zero_grad()
            calls.clear()
            output = model.infer_action(prompt=None, input_image=sample["video"][:1, :, 0],
                action_horizon=32, context=sample["context"][:1], context_mask=sample["context_mask"][:1],
                num_inference_steps=2, seed=42)
            self.assertEqual(len(calls), 1 + int(enabled))
            self.assertEqual(output["action"].shape, (32, 7))
            self.assertTrue(torch.isfinite(output["action"]).all())
        handle.remove()
        mask = model._register_attention_mask(19, 32, 4, torch.device("cpu"), noised_obs_length=4)
        self.assertTrue(mask[:15, :15].all())
        self.assertFalse(mask[:15, 15:19].any())
        self.assertFalse(mask[15:19, :15].any())
        self.assertTrue(mask[15:19, 15:19].all())
        self.assertFalse(mask[19:, :4].any())
        self.assertTrue(mask[19:, 4:7].all())
        self.assertFalse(mask[19:, 7:19].any())


if __name__ == "__main__":
    unittest.main()
