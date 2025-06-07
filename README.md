This is the README file for my thesis project.

**Thesis**: Deep Learning for Detection and Early Forecasting of Polar Lows Using High-Resolution Satellite Data

**TODO**:
- Improve Adversarial Erasing pipeline:
    - Load checkpoints corresponding to different iterations and see the activations using the adversarially-erased images. In particular, find the images with weird accumulated heatmaps and see if it is because the model thinks the image is "negative" in one of the iterations. I need to make the generate_heatmap() method also return the predicted class, so that if the predicted class is negative I save an all-transparent heatmap (all zeros).
