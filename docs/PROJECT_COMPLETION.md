# CIFAR-100 CNN Project Completion

## Objective

Finish the project with one controlled residual-learning ablation, final error analysis, and an evidence-based write-up. The goal is to explain **why** each architectural/training intervention helped or failed, not to continue tuning CIFAR-100 indefinitely.

## Current strongest verified result

| Model | Parameters | Best validation loss | Test accuracy |
|---|---:|---:|---:|
| 8-conv GAP CNN | 1.20M | 1.5357 | 59.89% |
| 10-conv GAP A | 4.77M | 1.6056 | 59.53% |
| 15-conv GAP | 7.91M | 1.6524 | 58.31% |
| 10-conv GAP B | 4.77M | 1.6552 | 59.57% |
| **SmallResNet** | **2.82M** | **1.4508** | **62.89%** |

SmallResNet is the provisional best model. The matched `PlainStageCNN` run is intentionally kept as the final controlled ablation before attributing the improvement specifically to residual shortcuts.

## Final controlled experiment

Compare `PlainStageCNN` and `SmallResNet` with the same:

- train/validation/test split
- stage widths
- blocks per stage
- stride-2 stage transitions
- BatchNorm pattern
- global average pooling
- initialization policy
- AdamW optimizer
- learning rate
- weight decay
- StepLR schedule
- augmentation
- epoch budget
- validation-loss checkpoint selection

The conceptual difference should be:

```text
Plain block:       y = F(x)
Residual block:    y = F(x) + shortcut(x)
```

Record:

| Metric | PlainStageCNN | SmallResNet |
|---|---:|---:|
| Parameters | pending | 2.82M |
| Best epoch | pending | 22 |
| Best validation loss | pending | 1.4508 |
| Peak validation accuracy | pending | 64.74% |
| Final training accuracy | pending | 99.91% |
| Final validation loss | pending | 2.1145 |
| Test accuracy at best-loss checkpoint | pending | 62.89% |

### Interpretation rules

- Worse **training** and validation metrics for the plain model support an optimization/degradation benefit from residual shortcuts.
- Similar training metrics but better validation for ResNet points more toward generalization.
- Similar or better plain performance means the earlier ResNet gain cannot be credited primarily to the shortcut.
- Do not select models using test accuracy.
- Do not substitute the peak-validation-accuracy checkpoint for the established minimum-validation-loss checkpoint unless the selection policy is changed for all compared runs.

## Final error analysis

After the matched comparison, run error analysis on the validation-loss-selected winning checkpoint:

```powershell
uv run python scripts/analyze_errors.py path\to\best_model.pt
```

The script writes `error_analysis.json` beside the checkpoint unless `--output` is supplied. Review:

1. Most frequent true-class -> predicted-class confusion pairs.
2. Weakest classes by recall.
3. Representative misclassified images using the existing visualization tooling.
4. Whether confusions are semantically/visually plausible.
5. Which errors appear limited by CIFAR-100's 32x32 resolution.

## Final ablation narrative

The final README/report should cover this progression:

```text
Baseline CNN
  -> pooling-frequency experiments
  -> GAP classifier
  -> initialization
  -> optimizer / LR scheduling
  -> weight decay
  -> augmentation
  -> BatchNorm
  -> deeper plain CNNs
  -> matched PlainStageCNN
  -> SmallResNet
  -> error analysis
```

Important lessons already supported by completed experiments:

- Repeated pooling confounded early depth experiments by destroying spatial resolution.
- GAP removed the large dense-head dependency on spatial dimensions.
- Scheduling reduced severe late degradation from a fixed learning rate.
- Weight decay produced modest regularization gains.
- Augmentation and BatchNorm were complementary.
- BatchNorm produced one of the largest improvements in the plain CNN sequence.
- The 8-conv staged GAP model improved strongly over the 4-conv model.
- Increasing plain GAP depth beyond that point did not improve validation loss under the established recipe.
- SmallResNet currently gives the best validation loss and test accuracy with fewer parameters than the deeper plain GAP models.
- SmallResNet still overfits heavily, so residual connections should not be described as an overfitting solution.

## Completion checklist

- [x] Baseline and architecture exploration completed
- [x] GAP, initialization, optimizer, scheduling, regularization, augmentation, and BatchNorm studied
- [x] Deeper plain GAP networks trained
- [x] SmallResNet implemented from scratch
- [x] SmallResNet trained and evaluated
- [x] Matched `PlainStageCNN` implemented
- [ ] Matched `PlainStageCNN` training completed
- [ ] Plain-vs-residual comparison recorded
- [ ] Winning checkpoint error analysis generated
- [ ] Top confusion pairs and weakest classes reviewed
- [ ] Final experiment table updated
- [ ] Final README conclusions updated

Once the unchecked items are complete, the educational objective of this CIFAR-100 project is complete. Further accuracy tuning should be treated as optional benchmarking, not a required project stage.
