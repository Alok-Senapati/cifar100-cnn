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
| **PlainStageCNN** | **2.78M** | **1.5242** | **59.47%** |
| **SmallResNet** | **2.82M** | **1.4508** | **62.89%** |

SmallResNet remains the best model after the matched control. With nearly the same parameter count and the same staged architecture/training recipe, the residual version improves best validation loss from `1.5242` to `1.4508` and test accuracy from `59.47%` to `62.89%`. This is the cleanest evidence in the project that skip connections materially improved the matched architecture.

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
| Parameters | 2,776,836 | 2.82M |
| Best epoch | 22 | 22 |
| Best validation loss | 1.5242 | 1.4508 |
| Validation accuracy at selected checkpoint | 59.60% | not recorded here |
| Test accuracy at best-loss checkpoint | 59.47% | 62.89% |
| Macro F1 at best-loss checkpoint | 59.17% | not recorded here |
| Peak validation accuracy | not supplied | 64.74% |
| Final training accuracy | not supplied | 99.91% |
| Final validation loss | not supplied | 2.1145 |

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

### Observed error patterns

The final SmallResNet checkpoint reaches **62.89% test accuracy**. Its largest errors are concentrated in visually or semantically similar classes:

| True class | Predicted class | Count |
|---|---|---:|
| girl | woman | 29 |
| maple_tree | oak_tree | 22 |
| otter | seal | 19 |
| streetcar | bus | 18 |
| man | woman | 18 |
| boy | woman | 17 |
| boy | girl | 15 |
| poppy | tulip | 14 |
| willow_tree | oak_tree | 14 |
| rose | tulip | 13 |
| woman | man | 13 |
| worm | snake | 13 |
| boy | man | 13 |
| pine_tree | oak_tree | 12 |
| hamster | mouse | 12 |

The weakest classes by recall are:

| Class | Recall |
|---|---:|
| otter | 21% |
| possum | 25% |
| boy | 32% |
| rabbit | 32% |
| lizard | 33% |
| girl | 34% |
| crocodile | 39% |
| lobster | 40% |
| squirrel | 40% |
| turtle | 40% |
| man | 43% |
| mouse | 46% |
| seal | 46% |
| shrew | 46% |
| bear | 48% |

These results point to a consistent failure mode: the model usually struggles where CIFAR-100 asks it to distinguish fine-grained categories that share shape, texture, pose, or context at only **32x32** resolution. Human subclasses (boy/girl/man/woman), tree species, flowers, small mammals, reptiles, and aquatic mammals dominate the error list. This suggests the remaining error is less about gross object detection and more about **fine-grained feature discrimination**.

A second pattern is asymmetric confusion. For example, many `girl` examples become `woman`, and several tree classes collapse into `oak_tree`. That suggests some learned classes act as stronger attractors in feature space than nearby alternatives. Future work could inspect classwise precision, calibration, embeddings, or representative misclassified images, but those are optional extensions rather than requirements for this project.

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
- The matched `PlainStageCNN` (2.78M parameters) reaches best validation loss `1.5242` and test accuracy `59.47%`.
- The matched `SmallResNet` (2.82M parameters) improves those to `1.4508` and `62.89%`, respectively, strongly supporting a benefit from residual shortcuts under the controlled staged comparison.
- This controlled result supports a residual-learning benefit, but it still does not mean residual connections solve overfitting; SmallResNet's late training reaches 99.91% train accuracy while validation loss rises to 2.1145.

## Completion checklist

- [x] Baseline and architecture exploration completed
- [x] GAP, initialization, optimizer, scheduling, regularization, augmentation, and BatchNorm studied
- [x] Deeper plain GAP networks trained
- [x] SmallResNet implemented from scratch
- [x] SmallResNet trained and evaluated
- [x] Matched `PlainStageCNN` implemented
- [x] Matched `PlainStageCNN` training completed
- [x] Plain-vs-residual comparison recorded
- [x] Winning checkpoint error analysis generated
- [x] Top confusion pairs and weakest classes reviewed
- [x] Final experiment table updated
- [x] Final README conclusions updated

All core checklist items are complete. The educational objective of this CIFAR-100 project is complete. Further work should be treated as optional benchmarking or research, not a required project stage.
