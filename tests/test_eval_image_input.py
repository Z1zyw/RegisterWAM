import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf


class EvalImageInputTest(unittest.TestCase):
    def test_raw_metadata_does_not_override_model_size(self):
        path = Path(__file__).resolve().parents[1] / "experiments/libero/eval_libero_single.py"
        tree = ast.parse(path.read_text())
        namespace = dict(torch=torch, np=np, Image=Image, DictConfig=object,
            RegisterWAMProcessor=object, get_libero_image=lambda obs: obs,
            _normalize_proprio=lambda *args: torch.zeros(1, 8), _extract_sim_state=lambda obs: None)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"_center_crop_resize", "_obs_to_model_input"}]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        processor = SimpleNamespace(shape_meta={"images": [{"shape": [3, 512, 512]}] * 2}, num_output_cameras=2)
        obs = {"image": np.zeros((512, 512, 3), dtype=np.uint8),
               "wrist_image": np.full((512, 512, 3), 255, dtype=np.uint8)}
        for direction, height, width in [("horizontal", 224, 448), ("vertical", 448, 224)]:
            cfg = OmegaConf.create({"data": {"train": {"concat_multi_camera": direction}}})
            x, proprio, _ = namespace["_obs_to_model_input"](obs, cfg, processor, width, height, "cpu", torch.float32)
            self.assertEqual(x.shape, (1, 3, height, width))
            self.assertEqual(proprio.shape, (1, 8))
            if direction == "horizontal":
                self.assertTrue((x[:, :, :, :224] == -1).all())
                self.assertTrue((x[:, :, :, 224:] == 1).all())
            else:
                self.assertTrue((x[:, :, :224] == -1).all())
                self.assertTrue((x[:, :, 224:] == 1).all())


if __name__ == "__main__":
    unittest.main()
