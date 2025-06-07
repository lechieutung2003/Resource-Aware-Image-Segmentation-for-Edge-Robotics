# Knowledge Distillation

## Methodology
This project uses **Logits-based Knowledge Distillation**.

1. **Teacher Logits**: The teacher network generates unnormalized logits for each pixel.
2. **Student Logits**: The student network also generates unnormalized logits.
3. **Temperature Scaling**: Both sets of logits are divided by a temperature parameter $T > 1$. This "softens" the probability distribution, revealing the relative confidence the teacher has in non-target classes (dark knowledge).
4. **KL Divergence Loss**: We compute the Kullback-Leibler divergence between the softened student probabilities and the softened teacher probabilities.
5. **Combined Objective**: The final loss is a weighted sum:
   $L = (1 - \alpha) \cdot L_{CE}(student, target) + \alpha \cdot L_{KD}(student, teacher)$

## Configuration
In `configs/config.yaml`, the following parameters control the distillation process:
- `temperature`: Scaling factor for softmax (default: 4.0)
- `alpha`: Weight of KD loss relative to CE loss (default: 0.5)

## Alternatives (Future Work)
- **Feature-map Distillation**: Matching intermediate feature maps between the teacher's ResNet and the student's MobileNet.

<!-- update -->

<!-- update -->

<!-- update -->

<!-- update -->

<!-- update -->

<!-- update -->
