# Evaluation

## Metrics
The evaluation script (`scripts/evaluate.py`) measures the following metrics:
- **Pixel Accuracy**: The percentage of pixels in the image that were correctly classified.
- **mean Intersection over Union (mIoU)**: The average of the IoU across all classes. IoU is calculated as `Intersection / Union`. It is the primary metric for semantic segmentation.

## Baseline Comparison
We compare three variations:
1. **Teacher**: Expected to have the highest mIoU but is too slow.
2. **Student Baseline**: MobileNetV3 student trained only on ground truth.
3. **Student Distilled (KD)**: MobileNetV3 student trained with knowledge distillation. Expected to close the gap between the baseline and the teacher.

## Visualization
During inference, visual results can be generated, overlaying the predicted mask on the original image, and comparing it against the ground truth and teacher prediction.

<!-- update -->

<!-- update -->

<!-- update -->
