"""Train RegisterWAM and evaluate the final checkpoint on LIBERO-plus."""
import argparse
import importlib.util
import json
import os
import random
from pathlib import Path
import subprocess
import sys
import time
from collections import defaultdict


def select_tasks(fraction, seed=42):
    from libero.libero import benchmark

    suites = ["libero_10", "libero_goal", "libero_spatial", "libero_object"]
    registry = benchmark.get_benchmark_dict()
    tasks = [(suite, task) for suite in suites for task in range(registry[suite]().n_tasks)]
    if fraction >= 1:
        return tasks
    path = Path(benchmark.__file__).resolve().parent / "task_classification.json"
    classification = json.loads(path.read_text()) if path.exists() else {}
    by_id = {(str(suite), int(item["id"])): item
             for suite, items in classification.items() for item in items if item.get("id") is not None}
    buckets = defaultdict(list)
    for suite, task in tasks:
        category = str(by_id.get((suite, task + 1), {}).get("category", "Unknown"))
        buckets[(suite, category)].append((suite, task))
    rng = random.Random(seed)
    chosen = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda item: item[1])
        count = min(len(bucket), max(1, int(round(len(bucket) * fraction))))
        chosen.extend(bucket[index] for index in sorted(rng.sample(range(len(bucket)), count)))
    rng.shuffle(chosen)
    return chosen


def persistent_chunks(tasks, gpus, workers_per_gpu):
    chunks = [[] for _ in range(len(gpus) * workers_per_gpu)]
    for index, task in enumerate(tasks):
        chunks[index % len(chunks)].append(task)
    return [(gpus[index % len(gpus)], index, chunk)
            for index, chunk in enumerate(chunks) if chunk]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train_eval", "train", "eval"], default="train_eval")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-root", default=".", help="Directory containing data/ and checkpoints/")
    parser.add_argument("--libero-root", help="LIBERO-plus checkout; defaults to the sibling of asset-root")
    parser.add_argument("--ckpt")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--task-fraction", type=float, default=0.2)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--max-tasks-per-gpu", type=int, default=3)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus) or args.max_tasks_per_gpu < 1 or args.num_trials < 1:
        parser.error("GPU IDs must be unique integers; worker count and trial count must be positive")
    if not 0 < args.task_fraction <= 1:
        parser.error("task-fraction must be in (0, 1]")
    libero_root = Path(args.libero_root).resolve() if args.libero_root else Path(args.asset_root).resolve().parent / "LIBERO-plus"
    if args.mode != "train":
        if not (libero_root / "libero/libero/benchmark/__init__.py").is_file():
            raise FileNotFoundError(f"LIBERO-plus checkout missing: {libero_root}; set --libero-root")
        sys.path.insert(0, str(libero_root))
        from libero.libero import benchmark
        print(f"[eval] benchmark={benchmark.__file__}", flush=True)
        registry = benchmark.get_benchmark_dict()
        counts = {suite: registry[suite]().n_tasks for suite in ["libero_10", "libero_goal", "libero_spatial", "libero_object"]}
        print(f"[eval] full_tasks={sum(counts.values())} suites={counts}", flush=True)
        if sum(counts.values()) == 40:
            raise RuntimeError("Loaded ordinary LIBERO (40 tasks), not LIBERO-plus; check --libero-root")
    jobs = select_tasks(args.task_fraction, seed=args.subset_seed) if args.mode != "train" else []
    if args.mode != "train":
        print(f"[eval] selected_tasks={len(jobs)}", flush=True)
    if args.mode != "train" and not args.dry_run and importlib.util.find_spec("libero") is None:
        raise ImportError("Install LIBERO-plus in this Python environment before starting train_eval")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(libero_root) + os.pathsep + str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["REGISTER_WAM_LIBERO_ROOT"] = str(libero_root)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["REGISTER_WAM_ASSET_ROOT"] = str(Path(args.asset_root).resolve())
    env.setdefault("MUJOCO_GL", "egl")
    train_cmd = [sys.executable, "-m", "accelerate.commands.launch",
                 "--config_file", "scripts/accelerate_configs/accelerate_zero1_ds.yaml",
                 "--num_processes", str(len(gpus)), "--num_machines", "1",
                 "scripts/train.py", "data=libero_plus", "task=libero_plus",
                 f"output_dir={output}", *overrides]
    if args.mode != "eval":
        print("[train]", " ".join(train_cmd), flush=True)
        if not args.dry_run:
            subprocess.run(train_cmd, cwd=root, env={**env, "CUDA_VISIBLE_DEVICES": args.gpus}, check=True)
    if args.mode == "train":
        return
    if args.dry_run:
        print(f"[eval] final checkpoint; GPUs={gpus}; tasks={len(jobs)}; trials={args.num_trials}; subset_seed={args.subset_seed}")
        print("[eval] tasks:", jobs)
        print("[eval] persistent workers:", persistent_chunks(jobs, gpus, args.max_tasks_per_gpu))
        return
    weights = sorted((output / "checkpoints/weights").glob("step_*.pt"))
    ckpt = Path(args.ckpt).resolve() if args.ckpt else (weights[-1] if weights else None)
    if ckpt is None or not ckpt.is_file():
        raise FileNotFoundError("No checkpoint found; use --ckpt or a completed training output")
    config = output / "config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"Training configuration missing: {config}")
    eval_dir = output / "eval_libero_plus" / time.strftime("%Y%m%d_%H%M%S")
    eval_dir.mkdir(parents=True, exist_ok=False)
    # Worker composes simulation settings with the exact saved training configuration.
    from omegaconf import OmegaConf
    simulation = OmegaConf.load(root / "configs/sim_libero.yaml")
    del simulation["defaults"]
    # Preserve the saved architecture, data paths and loss configuration.
    training = OmegaConf.load(config)
    model_eval = simulation.pop("model")
    cfg = OmegaConf.merge(training, simulation)
    cfg.model.merge_with(model_eval)
    cfg.ckpt = str(ckpt)
    cfg.EVALUATION.dataset_stats_path = str(output / "dataset_stats.json")
    cfg.EVALUATION.num_trials = args.num_trials
    cfg.EVALUATION.replan_steps = args.replan_steps
    cfg.EVALUATION.output_dir = str(eval_dir)
    cfg_file = eval_dir / "evaluation.yaml"
    OmegaConf.save(cfg, cfg_file)
    selected_tasks = [{"suite": suite, "task_id": task} for suite, task in jobs]
    (eval_dir / "tasks.json").write_text(json.dumps(selected_tasks, indent=2))
    running = []
    failures = []
    try:
        for gpu, slot, chunk in persistent_chunks(jobs, gpus, args.max_tasks_per_gpu):
            worker_dir = eval_dir / f"worker_gpu{gpu}_slot{slot}"
            worker_dir.mkdir()
            task_file = worker_dir / "tasks.txt"
            task_file.write_text("".join(f"{suite},{task}\n" for suite, task in chunk))
            log = (worker_dir / "worker.log").open("w")
            cmd = [sys.executable, "experiments/libero/eval_libero_persistent_worker.py",
                   "--config-path", str(eval_dir), "--config-name", "evaluation",
                   f"MULTIRUN.task_file={task_file}", f"EVALUATION.output_dir={worker_dir}", "gpu_id=0"]
            try:
                proc = subprocess.Popen(cmd, cwd=root, env={**env, "CUDA_VISIBLE_DEVICES": gpu}, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            running.append((gpu, proc, log, slot, chunk))
            print(f"[eval] persistent worker GPU={gpu} slot={slot} tasks={len(chunk)} log={worker_dir / 'worker.log'}", flush=True)
        while running:
            for item in list(running):
                gpu, proc, log, slot, chunk = item
                if proc.poll() is not None:
                    log.close()
                    running.remove(item)
                    if proc.returncode:
                        failures.append(f"GPU={gpu} slot={slot}")
                    print(f"[done] GPU={gpu} slot={slot} rc={proc.returncode}; workers_remaining={len(running)}", flush=True)
            if running:
                time.sleep(1)
    finally:
        for _, proc, log, _, _ in running:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, log, _, _ in running:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
    records = [json.loads(path.read_text()) for path in eval_dir.rglob("*_results.json")]
    successes = sum(row["successes"] for row in records)
    summary = {"checkpoint": str(ckpt), "selected_tasks": selected_tasks,
               "subset_seed": args.subset_seed, "completed_tasks": len(records), "failed_tasks": failures,
               "successes": successes, "episodes": len(records)*args.num_trials,
               "success_rate": successes/(len(records)*args.num_trials) if records else None}
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"Evaluation workers failed: {failures}; logs: {eval_dir}")


if __name__ == "__main__":
    main()
