# 🧠 How It Works

[← Back to README](../README.md)

The pipeline compresses a video clip into a single still image that encodes both the *shape* and *speed* of motion, then classifies that image with a shallow CNN suited to embedded deployment.

## The pipeline

### 1. 🎯 Tracking
Each frame is denoised (Gaussian blur + dilation/erosion), then the moving subject is separated from the background with **KNN-based background subtraction** — chosen for its robustness to shadows and changing lighting. Clips with no detectable moving contour are discarded.

### 2. ⏱️ Adaptive sampling
Instead of a fixed frame rate, the sampling interval adapts to how fast the subject is moving (estimated via optical flow):
- 🏃 Fast motion → sampled more densely, so nothing gets missed
- 🚶 Slow motion → sampled less densely, so nothing is redundant

The first and last few samples of each clip are trimmed to reduce noise.

### 3. 🖼️ Motion image encoding
Sampled frames are combined into a single image via **recency-weighted summation** — each new frame is folded in at full strength while everything accumulated so far decays by 30%, so recent motion dominates and older motion fades out.

This "dynamic image" is what gets classified — not a full video or an optical-flow volume — which is far cheaper for embedded hardware. Because it encodes a whole trajectory in one image, it naturally captures *where someone was headed* (toward the door vs. along it), not just what they looked like in one frame.

### 4. 🔄 Augmentation
Mirrored and randomly shifted variants of each motion image expand the training set with translation/orientation invariance.

### 5. 🧮 Classification
A shallow **AlexNet** (5 convolutional + 3 fully connected layers) with Tanh activations, Softmax output, and normal-Glorot weight initialization classifies the motion image into the trained set of classes (e.g. `entering` / `passing_by`).

Trained with SGD (tuned learning rate/momentum, favoring generalization over convergence speed) on a **60/30/10** train/validation/test split.

### 6. ⚡ Real-time inference
The same tracking → sampling → encoding pipeline runs end-to-end on a single freshly captured clip and feeds the trained model, returning the recognized intention by name — e.g. `"entering"` with a confidence score — for real-time/embedded use, such as triggering the door.

## 🏷️ About the class labels

Labels aren't hardcoded. Whatever label strings appear in your training data's `labeled_data.csv` become the classes the model learns and later reports by name (see `LabelEncoder` / `LabelMapRepository` in [Architecture](ARCHITECTURE.md)).

The `entering` / `passing_by` pair is just the reference scenario used throughout this repo's tests and demos.
