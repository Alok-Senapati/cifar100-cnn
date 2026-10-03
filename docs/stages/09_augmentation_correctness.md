# Stage 09 — Augmentation Correctness

## Objective

Ensure CUDA-side CIFAR-100 augmentation samples crop offsets and horizontal-flip decisions independently for each image rather than sharing one random decision across an entire batch.

## Why this stage exists

The training loop moves each batch to the target device before applying the optional training transform. That is the correct location for GPU augmentation. However, a batched torchvision transform call can share sampled random parameters across the batch. For the controlled CIFAR-100 experiments we want image-level stochastic augmentation.

## Change

`PerSampleBatchTransform` wraps the image transform and maps it across the batch after the tensor has moved to the device.

Training path:

```text
DataLoader
  -> float tensor [0, 1]
  -> move batch to CUDA
  -> per-image RandomCrop(32, padding=4)
  -> per-image RandomHorizontalFlip(p=0.5)
  -> Normalize
  -> model
```

Validation and test preprocessing remain deterministic.

## Control experiment

Re-run the current eight-convolution BN + augmentation model with the corrected augmentation implementation. Keep the established recipe fixed:

- architecture: current 8-convolution GAP CNN
- BatchNorm: enabled
- AdamW
- learning rate: 0.001
- weight decay: 1e-4
- StepLR: step size 20, gamma 0.5
- batch size: 512
- epochs: 150
- checkpoint criterion: minimum validation loss

This run becomes the clean plain-CNN reference for the residual-learning stage.

## Acceptance criteria

- Training transform is invoked once per image for a batched input.
- Validation/test transforms are unchanged.
- Existing training API remains compatible.
- Unit tests cover batch and single-image behavior.
