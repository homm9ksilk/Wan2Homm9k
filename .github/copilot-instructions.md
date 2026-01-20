# Wan2.2 Codebase Guide for AI Agents

## Project Overview

**Wan2.2** is a large-scale video generative model framework supporting multiple video generation tasks (T2V, I2V, TI2V, Speech2Video, Animation). The codebase uses a modular architecture where each generation task (T2V, I2V, S2V, Animate, TI2V) has a dedicated Python class that orchestrates text encoding, VAE encoding/decoding, and diffusion-based generation.

**Key architectural principle**: Task classes (e.g., `WanT2V`, `WanI2V`) encapsulate the full pipeline including model loading, sampling, and output generation. All tasks follow nearly identical initialization patterns with support for distributed training via FSDP and sequence parallelism.

## Critical Task Implementation Pattern

Every video generation task follows this structure in `wan/{task_name}.py`:

1. **Class initialization** loads three main components:
   - **Text Encoder**: T5 model (`T5EncoderModel`) with optional FSDP sharding
   - **VAE**: Either `Wan2_1_VAE` or `Wan2_2_VAE` for latent encoding/decoding
   - **DiT (Diffusion Transformer)**: Task-specific model (base `WanModel`, or `WanAnimateModel`, `WanModel_S2V`)

2. **Generate method** orchestrates:
   - Text/audio encoding to embeddings
   - Noise scheduling (via `FlowDPMSolverMultistepScheduler` or `FlowUniPCMultistepScheduler`)
   - Iterative denoising loop with cfg (classifier-free guidance)
   - VAE decoding to video frames
   - Video post-processing (fps, save to file)

3. **Distributed training support**: 
   - FSDP sharding available for T5 and DiT models
   - Sequence parallel strategy (`sp_attn_forward`, `sp_dit_forward`) for memory efficiency
   - Rank-aware initialization for multi-GPU setups

## Task-Specific Implementations

| Task | File | Config | VAE | Special Features |
|------|------|--------|-----|-----------------|
| **T2V** (Text-to-Video) | `text2video.py` | `wan_t2v_A14B.py` | Wan2_1 | Base diffusion + cfg, MoE architecture |
| **I2V** (Image-to-Video) | `image2video.py` | `wan_i2v_A14B.py` | Wan2_1 | Image conditioning, similar to T2V |
| **TI2V** (Text+Image-to-Video) | `textimage2video.py` | `wan_ti2v_5B.py` | Wan2_2 | Smaller 5B model, efficient 720p output |
| **S2V** (Speech-to-Video) | `speech2video.py` | `wan_s2v_14B.py` | Wan2_1 | Audio encoder (`AudioEncoder`), image input |
| **Animate** | `animate.py` | `wan_animate_14B.py` | Wan2_1 | Pose input, CLIPModel, face blocks, motion encoder, LoRA support |

## Configuration System

Configs in `wan/configs/` use `EasyDict` objects inheriting from `shared_config.py`:

- **Shared parameters**: T5 model type, bfloat16 dtype, frame_num (81), sample_fps (16), boundary threshold
- **Task-specific**: patch_size, dim (5120 for 14B), num_heads (40), num_layers (40), checkpoint paths
- **Sampling**: num_train_timesteps (1000), sample_steps (40), boundary (0.875), cfg guidance scales

When adding new models/configurations: extend shared config and import it in task-specific configs.

## Key Dependencies & Integration Points

- **Transformers**: T5 text encoder (UMT5-XXL by default)
- **Diffusers**: Scheduler classes, model base classes (ConfigMixin, ModelMixin)
- **Flash Attention**: Critical for performance; integrated in `modules/attention.py`
- **Accelerate/FSDP**: Multi-GPU training via `distributed/fsdp.py`
- **Decord**: Video frame decoding (used in S2V and Animate for input video reading)
- **SafeTensors**: Model checkpoint loading (used in S2V for audio encoder checkpoints)

## Developer Workflows

### Installation
```bash
# Ensure torch >= 2.4.0 first
pip install -r requirements.txt
pip install -r requirements_s2v.txt  # For S2V/speech synthesis (CosyVoice)
pip install -r requirements_animate.txt  # For Animate preprocessing
```

### Running Inference
```bash
# Main entry point: generate.py with task flag
python generate.py --task t2v-A14B --size '1280*720' --ckpt_dir ./models --prompt "..."
python generate.py --task animate-14B --video input.mp4 --pose pose.npy --mask mask.png
```

### Testing & Formatting
```bash
bash tests/test.sh <model_dir> <gpu_count>  # Run all inference tasks
make format  # isort + yapf on generate.py and wan/
```

### Key Debugging Flags in `generate.py`
- `--ckpt_dir`: Model checkpoint directory (required)
- `--rank`, `--local_rank`: For distributed inference setup
- `--t5_fsdp`, `--dit_fsdp`: Enable FSDP sharding for memory-constrained setups
- `--use_sp`: Enable sequence parallelism across GPUs

## Common Code Patterns

### 1. **Device & Dtype Management**
```python
self.device = torch.device(f"cuda:{device_id}")
self.param_dtype = config.param_dtype  # Usually bfloat16
```
Always respect `config.param_dtype` for model initialization to enable mixed precision.

### 2. **Checkpoint Loading Pattern** (in all task `__init__`)
```python
checkpoint_path = os.path.join(checkpoint_dir, config.checkpoint_name)
model = ModelClass(checkpoint_path=checkpoint_path, device=self.device)
```
Checkpoints are split by noise level (low_noise_checkpoint, high_noise_checkpoint) in T2V/I2V tasks.

### 3. **Sampling Loop Structure**
```python
for t in tqdm(timesteps):
    latents = self.denoise_step(latents, t, embeddings, cfg_scale)
```
Uses `retrieve_timesteps()` to manage diffusion timesteps, supports both scheduler types.

### 4. **Distributed Initialization**
```python
if t5_fsdp or dit_fsdp or use_sp:
    self.init_on_cpu = False  # Required for distributed strategies
shard_fn = partial(shard_model, device_id=device_id)
```
Pass `shard_fn` to model constructors only when FSDP is enabled.

## Integration with Diffusers/HuggingFace

Wan2.2 is integrated into diffusers; specialized diffusers implementations exist for:
- `Wan2.2-T2V-A14B-Diffusers` (in HF model hub)
- `Wan2.2-Animate-14B-Diffusers` (in diffusers PR #12526)

When modifying core models (`modules/model.py`), ensure compatibility with diffusers' `ModelMixin` and `ConfigMixin` base classes.

## File Structure Reference

- **`wan/`**: Main package
  - **`text2video.py`, `image2video.py`, etc.**: Task-specific generation classes
  - **`modules/model.py`**: Core DiT architecture (RoPE, attention, FFN)
  - **`modules/animate/`**: Pose-based animation components (CLIPModel, motion encoder, face blocks)
  - **`modules/s2v/`**: Speech-to-video components (AudioEncoder, audio processing)
  - **`distributed/`**: Multi-GPU utilities (FSDP, sequence parallelism)
  - **`utils/fm_solvers.py`**: Custom diffusion schedulers (Flow-based)
- **`generate.py`**: Main CLI entry point for all tasks
- **`configs/`**: Task configuration files using EasyDict

## Testing & Examples

Example prompts and inputs are provided in `generate.py` (EXAMPLE_PROMPT dict). See `examples/` folder for sample inputs (i2v_input.JPG, animate pose/video files).

Add unit tests in `tests/` when modifying core modules; current test structure runs inference on all models to validate integration.

---

**Last Updated**: January 2025 | Wan2.2 v2.2.0
