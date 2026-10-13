# RegisterWAM

Official code for **RegisterWAM**.

RegisterWAM connects video and action branches through learned register
tokens, built on Wan2.2-TI2V-5B and FastWAM. This repository provides
training and evaluation on LIBERO-plus.

## News

- **Coming soon:** Paper release.
- **Coming soon:** Public release of code and model checkpoints for LIBERO-plus.

## Installation

Use Python 3.10 and install a CUDA-compatible PyTorch build, then:

```bash
pip install -e ".[eval]"
```

Install LIBERO-plus and its simulation assets in the same environment.
Pass its checkout path with `--libero-root`.

## Model and Data Preparation

Prepare the Wan2.2 weights, interpolated ActionDiT backbone, LIBERO training
data, and cached text embeddings following
[FastWAM](https://github.com/yuantianyuan01/FastWAM). The asset directory
should contain:

```text
assets/
  checkpoints/Wan2.2-TI2V-5B/
  checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
  data/libero_mujoco3.3.2/
  data/text_embeds_cache/libero/
```

The dataset directory contains the `libero_10`, `libero_goal`, `libero_object`,
and `libero_spatial` datasets with the `_no_noops_lerobot` suffix.

## LIBERO-plus Configuration

- Model: [configs/model/register_wam_libero.yaml](configs/model/register_wam_libero.yaml).
- Data: [configs/data/libero_plus.yaml](configs/data/libero_plus.yaml).
- Training: [configs/task/libero_plus.yaml](configs/task/libero_plus.yaml).

Defaults: 8 registers, independent 5B video and 1B action experts, batched VAE,
32-step actions, two 224x224 cameras, batch size 16 per GPU, 10 epochs,
learning rate 1e-4, and 5% warmup. Loss coefficients are action 0.2, video 1,
and noised-observation auxiliary 2; pure-noise video loss has weight 2.
`visual_attention_mode: bidirectional` lets observation, register, and future
tokens attend to each other. The noised-observation auxiliary stream is
isolated. Action tokens attend only to registers and other action tokens.
Both experts, registers, and the proprioception encoder use the same learning
rate; the VAE and text encoder remain frozen.

## Training and Evaluation

Run from the repository root. This trains on eight GPUs and then evaluates
all LIBERO-plus tasks, using three persistent workers per GPU:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/train_eval_libero_plus.py --mode train_eval --asset-root /path/to/assets --libero-root /path/to/LIBERO-plus --gpus 0,1,2,3,4,5,6,7 --max-tasks-per-gpu 3 --task-fraction 1 --num-trials 1 --output-dir runs/libero_plus
```

Use `--mode train` for training only. To evaluate a trained checkpoint:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/train_eval_libero_plus.py --mode eval --asset-root /path/to/assets --libero-root /path/to/LIBERO-plus --gpus 0,1,2,3,4,5,6,7 --task-fraction 1 --output-dir runs/libero_plus --ckpt runs/libero_plus/checkpoints/weights/step_021700.pt
```

Evaluation uses the run's saved configuration and normalization statistics.
Defaults are 10 inference steps and `--replan-steps 10`. Results are saved in
`runs/libero_plus/eval_libero_plus/<timestamp>/`. Set `--task-fraction 0.2`
for a stratified subset; `--subset-seed` defaults to 42. Hydra training
overrides such as `num_epochs=8` can be appended to the training command.

## Acknowledgements

This implementation builds on FastWAM, Wan2.2, and LIBERO-plus. See
[LICENSE](LICENSE) for the repository license.
