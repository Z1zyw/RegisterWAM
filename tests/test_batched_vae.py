import unittest
from types import SimpleNamespace

import torch

from register_wam.models.wan22.wan_video_vae import WanVideoVAE


class BatchedEncodeTest(unittest.TestCase):
    def test_tensor_and_sequence_use_one_core_call(self):
        calls = []

        def encode(batch, device):
            calls.append(batch.shape)
            return batch.to(device) * 2

        vae = SimpleNamespace(single_encode=encode)
        batch = torch.randn(3, 3, 9, 8, 8)
        for inputs in (batch, list(batch)):
            output = WanVideoVAE.encode(vae, inputs, device="cpu")
            torch.testing.assert_close(output, batch * 2)
        self.assertEqual(calls, [batch.shape, batch.shape])


if __name__ == "__main__":
    unittest.main()
