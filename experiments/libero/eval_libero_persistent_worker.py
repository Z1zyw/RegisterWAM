import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
libero_root = os.environ.get("REGISTER_WAM_LIBERO_ROOT")
if libero_root:
    sys.path.insert(0, libero_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import experiments.libero.eval_libero_single as eval_libero_single  # noqa: E402
from experiments.libero.eval_libero_single import (  # noqa: E402
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    run_single_task,
)
from register_wam.datasets.lerobot.processors.register_wam_processor import RegisterWAMProcessor  # noqa: E402
from register_wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from register_wam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
import libero.libero.envs.bddl_utils as BDDLUtils  # noqa: E402


def _register_resolver_if_needed(name, fn):
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, fn)


_register_resolver_if_needed("eval", eval)
_register_resolver_if_needed("max", lambda x: max(x))
_register_resolver_if_needed("split", lambda s, idx: s.split("/")[int(idx)])


_ORIG_TORCH_LOAD = torch.load


def _patch_torch_load_for_libero() -> None:
    def _torch_load_compat(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _ORIG_TORCH_LOAD(*args, **kwargs)

    torch.load = _torch_load_compat


def _get_task_description_for_prompt(task, mode: str) -> str:
    mode = str(mode or "raw")
    if mode == "raw":
        return task.language
    if mode in {"bddl_language", "clean_bddl", "clean_non_language"}:
        task_bddl_file = (
            Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        try:
            problem_info = BDDLUtils.get_problem_info(str(task_bddl_file))
            language = str(problem_info.get("language_instruction", "")).strip()
            if language:
                return language
        except Exception as exc:
            print(
                f"[persistent-worker][prompt-mode] failed to read BDDL language "
                f"for {task_bddl_file}: {exc}; fallback to task.language"
            )
        return task.language
    raise ValueError(
        f"Unsupported EVALUATION.libero_plus_prompt_mode={mode!r}. "
        "Supported modes: raw, bddl_language."
    )


def _patch_get_libero_env_path_compat(prompt_mode: str = "raw") -> None:
    orig_get_libero_env = eval_libero_single.get_libero_env

    def _wrapped_get_libero_env(task, resolution, seed, env_num=1):
        env, task_description = orig_get_libero_env(task, resolution, seed, env_num=env_num)
        return env, task_description

    # Patch the helper inside eval_libero_single via its source module dependency.
    import experiments.libero.libero_utils as libero_utils

    def _compat_get_libero_env(task, resolution, seed, env_num=1):
        task_description = _get_task_description_for_prompt(task, prompt_mode)
        task_bddl_file = (
            Path(libero_utils.get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env_args = {
            "bddl_file_name": str(task_bddl_file),
            "camera_heights": resolution,
            "camera_widths": resolution,
        }
        if env_num > 1:
            env = libero_utils.SubprocVectorEnv([
                lambda: libero_utils.OffScreenRenderEnv(**env_args) for _ in range(env_num)
            ])
        else:
            env = libero_utils.OffScreenRenderEnv(**env_args)
        env.seed(seed)
        return env, task_description

    libero_utils.get_libero_env = _compat_get_libero_env
    eval_libero_single.get_libero_env = _compat_get_libero_env


def _load_task_list(task_file: Path) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    with task_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            suite, task_id = line.split(",", 1)
            tasks.append((suite.strip(), int(task_id.strip())))
    return tasks


def _prepare_runtime(cfg: DictConfig):
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: RegisterWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    return {
        "model": model,
        "processor": processor,
        "action_horizon": action_horizon,
        "input_h": input_h,
        "input_w": input_w,
        "model_device": model_device,
        "dataset_stats_path": str(dataset_stats_path),
    }


def _run_one_task(
    *,
    cfg: DictConfig,
    runtime: dict,
    suite_name: str,
    task_id: int,
    worker_output_dir: Path,
    worker_gpu_id: int,
) -> dict:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    suite_video_dir = worker_output_dir / suite_name / "videos"
    suite_video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = worker_output_dir / suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = {
        "task_suite": suite_name,
        "task_id": task_id,
        "task_name": str(task.name),
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(worker_gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=runtime["model"],
        processor=runtime["processor"],
        cfg=cfg,
        video_dir=suite_video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=runtime["action_horizon"],
        input_w=runtime["input_w"],
        input_h=runtime["input_h"],
        model_device=runtime["model_device"],
    )
    results.update(task_results)
    results["duration"] = time.time() - start_time

    suite_output_dir = worker_output_dir / suite_name
    suite_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = suite_output_dir / f"gpu{worker_gpu_id}_task{task_id}_results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"[persistent-worker] suite={suite_name} task={task_id} "
        f"success={results['successes']}/{cfg.EVALUATION.num_trials} "
        f"time={results['duration']:.2f}s"
    )
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero")
def main(cfg: DictConfig):
    partial_state = PartialState()
    partial_state.config = cfg

    task_file_cfg = cfg.MULTIRUN.get("task_file")
    if task_file_cfg is None:
        raise ValueError("MULTIRUN.task_file must be set for persistent worker mode.")
    task_file = Path(os.path.expanduser(os.path.expandvars(str(task_file_cfg))))
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    tasks = _load_task_list(task_file)
    if len(tasks) == 0:
        raise ValueError(f"Task file is empty: {task_file}")

    worker_output_dir = Path(cfg.EVALUATION.output_dir)
    worker_output_dir.mkdir(parents=True, exist_ok=True)
    worker_gpu_id = int(cfg.gpu_id)

    logging.info("Persistent worker starting: gpu_id=%s task_file=%s num_tasks=%s", worker_gpu_id, task_file, len(tasks))
    _patch_torch_load_for_libero()
    prompt_mode = str(cfg.EVALUATION.get("libero_plus_prompt_mode", "raw"))
    _patch_get_libero_env_path_compat(prompt_mode=prompt_mode)
    runtime = _prepare_runtime(cfg)

    summary = {
        "task_file": str(task_file),
        "gpu_id": worker_gpu_id,
        "ckpt": str(cfg.ckpt),
        "dataset_stats_path": runtime["dataset_stats_path"],
        "num_tasks": len(tasks),
        "libero_plus_prompt_mode": prompt_mode,
        "tasks": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    total_start = time.time()
    for suite_name, task_id in tasks:
        result = _run_one_task(
            cfg=cfg,
            runtime=runtime,
            suite_name=suite_name,
            task_id=task_id,
            worker_output_dir=worker_output_dir,
            worker_gpu_id=worker_gpu_id,
        )
        summary["tasks"].append(
            {
                "task_suite": suite_name,
                "task_id": task_id,
                "successes": int(result["successes"]),
                "total_episodes": int(result["total_episodes"]),
                "duration": float(result["duration"]),
            }
        )

    summary["duration"] = time.time() - total_start
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary_file = worker_output_dir / f"worker_gpu{worker_gpu_id}_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print(
        f"[persistent-worker] done gpu_id={worker_gpu_id} "
        f"num_tasks={len(tasks)} total_time={summary['duration']:.2f}s"
    )


if __name__ == "__main__":
    main()
