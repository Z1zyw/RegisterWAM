import types
import unittest

import torch

from register_wam.models.wan22.register_wam import RegisterWAM


class TrainingNoiseAlignmentTest(unittest.TestCase):
    def test_rng_matches_legacy_for_pure_and_sample_future(self):
        for mode in ("pure_noise", "sample"):
            for enabled in (False, True):
                with self.subTest(mode=mode, noised_obs=enabled):
                    video = torch.zeros(2, 2, 3, 4, 4)
                    action = torch.zeros(2, 32, 7)
                    calls = []

                    def sample_t(batch_size, device, dtype):
                        return torch.rand(batch_size, device=device, dtype=dtype) * 1000

                    def add_noise(clean, noise, timestep):
                        calls.append((noise.clone(), timestep.clone()))
                        return noise.clone()

                    scheduler = types.SimpleNamespace(
                        num_train_timesteps=1000, sample_training_t=sample_t,
                        add_noise=add_noise,
                        training_target=lambda clean, noise, timestep: noise - clean,
                    )

                    def stop_before_forward(**kwargs):
                        raise RuntimeError("stop before forward")

                    stub = types.SimpleNamespace(
                        device=torch.device("cpu"), future_noise_mode=mode,
                        noised_obs_enabled=enabled, noised_obs_timestep=500,
                        include_first_obs_video_loss=True,
                        train_video_scheduler=scheduler, train_action_scheduler=scheduler,
                        video_expert=types.SimpleNamespace(pre_dit=stop_before_forward),
                        build_inputs=lambda sample, tiled: {
                            "input_latents": video, "action": action,
                            "first_frame_latents": video[:, :, :1],
                            "context": None, "context_mask": None,
                            "fuse_vae_embedding_in_latents": True,
                        },
                    )
                    torch.manual_seed(42)
                    expected = [(torch.randn_like(video),
                                 sample_t(2, "cpu", video.dtype) if mode == "sample"
                                 else torch.full((2,), 1000.0))]
                    if enabled:
                        expected.append((torch.randn_like(video[:, :, :1].contiguous()),
                                         torch.full((2,), 500.0)))
                    expected.append((torch.randn_like(action), sample_t(2, "cpu", action.dtype)))
                    expected_rng = torch.get_rng_state()
                    torch.manual_seed(42)
                    with self.assertRaisesRegex(RuntimeError, "stop before forward"):
                        RegisterWAM.training_loss(stub, {})
                    self.assertEqual(len(calls), len(expected))
                    for actual, reference in zip(calls, expected):
                        for a, b in zip(actual, reference):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))


if __name__ == "__main__":
    unittest.main()
