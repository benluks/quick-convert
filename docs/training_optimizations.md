# Training Optimizations

All optimizations are controlled via Hydra overrides or by editing `configs/trainer/controllable_rvq.yaml`.

---

## Mixed Precision

Set `trainer.precision` to reduce memory usage and unlock flash attention kernels.

```yaml
# configs/trainer/controllable_rvq.yaml
precision: bf16-mixed   # recommended (bf16, stable on Ampere+)
precision: 16-mixed     # fp16 alternative
precision: null         # disabled (default, uses train_kwargs.precision = 32)
```

**Command-line:**
```
train train_controllable_rvq_librispeech trainer.precision=bf16-mixed
```

---

## Flash Attention

Enabled by default on all conformer blocks. Requires CUDA and mixed precision to dispatch the flash kernel.

```yaml
# configs/components/encoders/parallel_conformer.yaml
use_flash_attention: true   # default
use_flash_attention: false  # fall back to manual attention
```

**Command-line:**
```
train train_controllable_rvq_librispeech trainer.module.encoder.use_flash_attention=false
```

---

## torch.compile

Compiles selected submodules before training begins.

```yaml
# configs/trainer/controllable_rvq.yaml
compile:
  enabled: true
  targets: [encoder]        # start with encoder only; add decoder once stable
  backend: inductor
  mode: default             # or reduce-overhead / max-autotune
  fullgraph: false          # allow graph breaks for robustness
  dynamic: true             # handle variable-length batches
```

**Command-line:**
```
train train_controllable_rvq_librispeech trainer.compile.enabled=true trainer.compile.targets=[encoder]
```

---

## Multi-GPU (DDP)

Enables PyTorch DDP via Lightning. Batch size and learning rate should be scaled accordingly.

```yaml
# configs/trainer/controllable_rvq.yaml
ddp:
  enabled: true
  accelerator: gpu
  devices: auto       # or an integer, e.g. 4
  num_nodes: 1
  strategy: ddp
  sync_batchnorm: false
```

**Command-line:**
```
train train_controllable_rvq_librispeech trainer.ddp.enabled=true pipeline.train_kwargs.devices=4
```

---

## cuDNN Benchmark

Speeds up convolution ops when input shapes are mostly stable across batches.

```yaml
# configs/trainer/controllable_rvq.yaml
cudnn_benchmark: true   # enable
cudnn_benchmark: null   # leave unchanged (default)
```

---

## Recommended Combination

For a full-speed multi-GPU run with flash attention:

```
train train_controllable_rvq_librispeech \
  trainer.precision=bf16-mixed \
  trainer.ddp.enabled=true \
  trainer.compile.enabled=true \
  trainer.compile.targets=[encoder] \
  trainer.cudnn_benchmark=true
```
