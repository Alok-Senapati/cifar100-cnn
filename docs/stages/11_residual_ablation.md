# Stage 11 — Controlled Residual Ablation

## Objective

Separate the effect of skip connections from unrelated architectural differences.

The earlier eight-convolution plain CNN uses max-pooling transitions, while `SmallResNet` downsamples with stride-2 convolutions. This stage introduces `PlainStageCNN`, which matches the residual model's stem, stage widths, number of two-convolution blocks, stride-2 transitions, BatchNorm usage, GAP head, initialization strategy, and training recipe—but removes the residual addition.

## Models to compare

### Plain staged control

```text
Stem -> [PlainBlock x2] -> [PlainBlock x2, stride 2 at entry]
     -> [PlainBlock x2, stride 2 at entry]
     -> [PlainBlock x2, stride 2 at entry]
     -> GAP -> Linear
```

### Residual model

Same stage structure, but each two-convolution block returns:

```text
ReLU(F(x) + shortcut(x))
```

## Controlled variables

Keep identical across both runs:

- stage channels: 32, 64, 128, 256
- two blocks per stage
- BatchNorm
- custom initialization
- AdamW
- LR 0.001
- weight decay 1e-4
- StepLR(step_size=20, gamma=0.5)
- batch size 512
- corrected per-image crop + flip augmentation
- 150-epoch budget
- same train/validation split
- checkpoint selection by minimum validation loss

## What to record

| Metric | Plain staged CNN | Small ResNet |
|---|---:|---:|
| Parameters | | |
| Best epoch | | |
| Best validation loss | | |
| Validation accuracy at best-loss checkpoint | | |
| Final training accuracy | | |
| Final validation loss | | |
| Test accuracy from best-loss checkpoint | | |

## Interpretation

- If ResNet trains deeper/equivalent capacity more effectively and improves validation loss, that supports the optimization benefit of residual learning.
- If train accuracy improves but validation does not, the issue is generalization rather than optimization.
- If both models behave similarly, do not claim skip connections were decisive from this experiment.

The goal is an interpretable ablation, not a benchmark-winning CIFAR-100 score.
