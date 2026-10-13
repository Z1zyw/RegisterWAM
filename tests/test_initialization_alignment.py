import unittest

import torch
from torch import nn

from register_wam.models.wan22.register_wam import RegisterWAM


class InitializationAlignmentTest(unittest.TestCase):
    def test_register_and_proprio_match_legacy_initialization(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)
                expected_proprio = nn.Linear(8, 16).to(dtype)
                expected_register = (torch.randn(1, 24, 32) / (32 ** 0.5)).to(dtype)
                expected_rng = torch.get_rng_state()
                video = nn.Identity()
                video.hidden_dim = 32
                action = nn.Identity()
                mot = nn.Module()
                mot.mixtures = nn.ModuleDict({"video": video, "action": action})
                torch.manual_seed(42)
                model = RegisterWAM(
                    video, action, mot, nn.Identity(), text_dim=16,
                    proprio_dim=8, register_num_tokens=24, torch_dtype=dtype,
                )
                torch.testing.assert_close(model.mot.register_tokens, expected_register, rtol=0, atol=0)
                for actual, expected in zip(model.proprio_encoder.parameters(), expected_proprio.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))


if __name__ == "__main__":
    unittest.main()
