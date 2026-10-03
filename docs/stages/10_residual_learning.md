# Stage 10 — Residual Learning

## Objective

Replace the plain staged CNN with a small CIFAR-style residual network implemented from scratch and compare optimization/generalization under the same training recipe.

## Architecture

```text
Stem: 3x3 Conv -> BN -> ReLU

Stage 1: 32 channels, 2 residual blocks, stride 1
Stage 2: 64 channels, 2 residual blocks, stride 2 at entry
Stage 3: 128 channels, 2 residual blocks, stride 2 at entry
Stage 4: 256 channels, 2 residual blocks, stride 2 at entry

AdaptiveAvgPool2d(1)
Linear(256, 100)
```

A residual block computes:

```text
F(x) = Conv -> BN -> ReLU -> Conv -> BN
output = ReLU(F(x) + shortcut(x))
```

The shortcut is identity when shape is unchanged and a learned 1x1 projection when channels or spatial resolution change.

## Why no ImageNet stem

CIFAR-100 images are 32x32. A 7x7 stride-2 stem followed by max pooling would remove too much spatial resolution before useful features are learned, so this implementation uses a 3x3 stride-1 stem.

## Controlled training recipe

Keep the established recipe fixed:

- AdamW
- LR 0.001
- weight decay 1e-4
- StepLR(step_size=20, gamma=0.5)
- batch size 512
- 150 epochs
- training-only crop + flip augmentation
- custom initialization
- checkpoint selection by minimum validation loss

## Experiment question

Does residual learning make a deeper staged network easier to optimize and/or improve validation performance compared with the corrected plain-CNN control?

## Metrics to compare

- parameter count
- best epoch
- best validation loss
- validation accuracy at selected checkpoint
- training accuracy at selected checkpoint
- final training accuracy
- final validation loss
- test accuracy of the selected checkpoint
- gradient norms / stability

## Caveat

The existing eight-convolution plain network uses max-pooling stage transitions while this residual network uses stride-2 convolutions. This first ResNet run is therefore an architectural milestone, not yet a one-variable residual ablation. Stage 11 adds the cleaner comparison and ablation analysis.
