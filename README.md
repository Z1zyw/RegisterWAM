# RegisterWAM

Official code for **RegisterWAM**.

Training and evaluation on LIBERO-plus.

## News

- **Coming soon:** Paper release.
- **Coming soon:** Public release of code and model checkpoints for LIBERO-plus.

## Installation

Python 3.10 and a CUDA-capable NVIDIA GPU are required. The commands below
install PyTorch with CUDA 12.8; use a compatible NVIDIA driver.

```bash
conda create -n register-wam python=3.10 -y
conda activate register-wam
git clone https://github.com/Z1zyw/RegisterWAM.git
cd RegisterWAM
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install "numpy==1.26.4" "transformers==4.49.0" "datasets==3.6.0"
pip install -e ".[eval]"
```

### LIBERO-plus

Set up [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) and its
simulation assets following the official
[installation guide](https://github.com/sylvestf/LIBERO-plus#installation).
Use the same Python environment and pass its checkout path with `--libero-root`.
Do not overwrite this environment with the legacy dependency versions in
LIBERO-plus's `requirements.txt`.

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
`--asset-root` points to the directory containing `checkpoints/` and `data/`.
Keep the complete Wan2.2 checkpoint directory, including the VAE, text encoder,
and tokenizer required for evaluation.

## LIBERO-plus Configuration

- Model: [configs/model/register_wam_libero.yaml](configs/model/register_wam_libero.yaml).
- Data: [configs/data/libero_plus.yaml](configs/data/libero_plus.yaml).
- Training: [configs/task/libero_plus.yaml](configs/task/libero_plus.yaml).

## Training and Evaluation

Run from the repository root. This trains on eight GPUs and then evaluates
all LIBERO-plus tasks, using three persistent workers per GPU:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/train_eval_libero_plus.py --mode train_eval --asset-root /path/to/assets --libero-root /path/to/LIBERO-plus --gpus 0,1,2,3,4,5,6,7 --max-tasks-per-gpu 3 --task-fraction 1 --num-trials 1 --output-dir runs/libero_plus
```

Use `--mode train` for training only. To evaluate the latest checkpoint:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/train_eval_libero_plus.py --mode eval --asset-root /path/to/assets --libero-root /path/to/LIBERO-plus --gpus 0,1,2,3,4,5,6,7 --task-fraction 1 --output-dir runs/libero_plus
```

For evaluation, `--output-dir` must contain the training run's `config.yaml`,
`dataset_stats.json`, and checkpoints. Use `--ckpt /path/to/step_XXXXXX.pt`
to select a specific checkpoint. Results are saved in
`runs/libero_plus/eval_libero_plus/<timestamp>/`. Set `--task-fraction 0.2`
for a stratified subset. Append Hydra overrides to the training command,
for example `num_epochs=8`.

## Acknowledgements

This implementation builds on FastWAM, Wan2.2, and LIBERO-plus. See
[LICENSE](LICENSE) for the repository license.
