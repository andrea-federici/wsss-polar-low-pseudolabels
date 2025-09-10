This is the README file for my thesis project.

**Thesis**: Deep Learning for Detection and Early Forecasting of Polar Lows Using High-Resolution Satellite Data

### Multi-label Training

The training pipeline supports multi-label classification. When working with
datasets such as Pascal VOC, enable multi-label mode in the training
configuration:

```yaml
# conf/training/pascal_voc.yaml
criterion: bce_with_logits
multi_label: true
threshold: 0.5  # prediction threshold for positive labels
```

The recommended loss for this setting is `BCEWithLogitsLoss`, and predictions are
obtained by applying a sigmoid and comparing against the specified threshold.