# Stage 12 — Error Analysis and Project Completion

## Objective

Finish the CIFAR-100 project with evidence-driven error analysis, a compact ablation summary, and a final project narrative.

## Required final comparisons

At minimum, include these milestones:

| Model / intervention | Main question |
|---|---|
| Original baseline CNN | What does a simple CNN achieve? |
| Reduced pooling | Does retaining spatial resolution help by itself? |
| GAP head | Does removing the large dense head improve generalization? |
| Width / depth exploration | Where does capacity help? |
| Custom initialization | Does signal-aware initialization improve optimization? |
| AdamW + scheduler | Does optimization strategy improve late training? |
| Weight decay | Does explicit regularization reduce overfitting? |
| Crop + flip augmentation | Does invariance-based data augmentation help? |
| BatchNorm | Does normalized intermediate activation materially improve trainability? |
| 8-convolution plain CNN | Does additional staged depth improve representation? |
| Matched plain staged CNN | Residual-ablation control |
| SmallResNet | Do skip connections improve optimization/generalization? |

Select models using the established validation-loss checkpoint criterion. Do not retrospectively choose by test accuracy.

## Error analysis

Run:

```powershell
uv run python scripts/analyze_errors.py <path-to-best_model.pt>
```

The generated `error_analysis.json` records:

- selected checkpoint test accuracy
- most frequent true-class -> predicted-class confusions
- weakest classes by classwise accuracy

Use the output to answer:

1. Which CIFAR-100 classes are most frequently confused?
2. Are the errors semantically or visually plausible?
3. Which classes have the weakest recall?
4. Which failures are likely limited by 32x32 resolution?
5. Does the final model still show systematic class-specific weaknesses?

## Final project conclusions

The final write-up should distinguish observations from interpretation. In particular:

- Do not claim residual connections caused an improvement unless the matched Stage 11 control supports it.
- Do not use the test set to select hyperparameters.
- Report that parameter-count changes confound earlier width/depth comparisons where applicable.
- Report that the original depth experiment also changed downsampling frequency.

## Completion checklist

- [ ] Corrected per-image augmentation control completed
- [ ] SmallResNet trained from scratch
- [ ] Matched plain staged control trained
- [ ] Residual ablation table completed
- [ ] Best checkpoint selected by validation loss
- [ ] Error analysis JSON generated
- [ ] Top confusion pairs reviewed
- [ ] Weakest classes reviewed
- [ ] Final confusion matrix reviewed
- [ ] Final experiment table added to project documentation
- [ ] README updated with architecture progression and key lessons
- [ ] Final model can be explained without relying on torchvision ResNet implementation

Once these items are complete, the educational objective of the project is finished. Further CIFAR-100 tuning should be treated as optional benchmarking rather than core project work.
