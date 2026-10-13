import ast
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class PersistentEvalTest(unittest.TestCase):
    def test_round_robin_uses_all_gpus_and_respects_slots(self):
        spec = importlib.util.spec_from_file_location("launcher", ROOT / "scripts/train_eval_libero_plus.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        tasks = [("suite", i) for i in range(40)]
        chunks = launcher.persistent_chunks(tasks, [str(i) for i in range(8)], 2)
        self.assertEqual(len(chunks), 16)
        self.assertEqual([gpu for gpu, _, _ in chunks[:8]], [str(i) for i in range(8)])
        self.assertEqual(sorted(task for _, _, chunk in chunks for task in chunk), tasks)
        self.assertEqual(len(launcher.persistent_chunks(tasks[:8], [str(i) for i in range(8)], 2)), 8)

    def test_worker_prepares_runtime_once_for_multiple_tasks(self):
        # Run the original worker control flow without loading a 5B model or simulator.
        tree = ast.parse((ROOT / "experiments/libero/eval_libero_persistent_worker.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        function.decorator_list = []
        prepared, evaluated = [], []

        def prepare(cfg):
            prepared.append(cfg)
            return {"dataset_stats_path": "stats.json"}

        def run(**kwargs):
            evaluated.append((kwargs["suite_name"], kwargs["task_id"]))
            return {"successes": 1, "total_episodes": 1, "duration": 0.1}

        with tempfile.TemporaryDirectory() as directory:
            tasks = Path(directory) / "tasks.txt"
            tasks.write_text("suite,0\nsuite,1\n")
            namespace = dict(DictConfig=object, PartialState=SimpleNamespace, Path=Path,
                os=__import__("os"), time=time, json=json, logging=__import__("logging"),
                NumpyEncoder=json.JSONEncoder, _load_task_list=lambda _: [("suite", 0), ("suite", 1)],
                _patch_torch_load_for_libero=lambda: None,
                _patch_get_libero_env_path_compat=lambda **_: None,
                _prepare_runtime=prepare, _run_one_task=run)
            exec(compile(ast.Module(body=[function], type_ignores=[]), "worker", "exec"), namespace)
            cfg = SimpleNamespace(MULTIRUN={"task_file": str(tasks)},
                EVALUATION=SimpleNamespace(output_dir=directory, get=lambda *args: "raw"),
                gpu_id=0, ckpt="weights.pt")
            namespace["main"](cfg)
            self.assertEqual(len(prepared), 1)
            self.assertEqual(evaluated, [("suite", 0), ("suite", 1)])


if __name__ == "__main__":
    unittest.main()
