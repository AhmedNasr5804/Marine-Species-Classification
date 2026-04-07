"""
transforms.py — Image preprocessing pipelines for underwater imagery.

Design decisions
----------------
- Input size: 224×224 — the standard for ImageNet-pretrained backbones (ResNet-50,
  EfficientNet-B0, MobileNetV2). Using the same size for Model A keeps comparisons fair.

- Normalisation with ImageNet mean/std: transfer-learning backbones were pretrained on
  ImageNet with these statistics. Using the same normalisation ensures the pretrained
  weight distributions are not perturbed at fine-tuning time. For Model A (trained from
  scratch) it also helps optimiser stability by keeping pixel values in a similar range.

- Colour jitter (training only): underwater imagery suffers from wavelength-dependent
  light attenuation — red channels attenuate fastest, giving images a blue-green cast
  that varies with depth and water clarity. Mild brightness/contrast/saturation jitter
  teaches the model to be invariant to these depth-induced colour shifts.
  hue jitter is kept very small (0.05) to avoid producing unrealistic colours.

- Random horizontal flip (training only): marine organisms have no preferred orientation
  relative to the camera, so horizontal mirroring is a valid augmentation that doubles
  the effective dataset size.

- Random resized crop (training only): introduces scale and positional invariance without
  distorting aspect ratios too aggressively. Lower scale bound of 0.7 avoids cropping
  so tightly that the organism is barely visible.

- Val/test transform: deterministic resize + centre crop + normalise only — no stochastic
  augmentation, so metrics are reproducible across runs.
"""

import torchvision.transforms as T

# ImageNet normalisation statistics (mean and std per channel, RGB order).
# Source: https://pytorch.org/vision/stable/models.html
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224  # pixels; change here to propagate everywhere


def get_train_transform(img_size: int = IMG_SIZE) -> T.Compose:
    """Augmented pipeline applied to training images only."""
    return T.Compose([
        # Scale then crop: provides scale/translation invariance.
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        # Horizontal flip: valid symmetry for underwater organisms.
        T.RandomHorizontalFlip(p=0.5),
        # Colour jitter: compensates for depth-dependent water colour cast.
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(img_size: int = IMG_SIZE) -> T.Compose:
    """Deterministic pipeline applied to validation, test, and OOD images."""
    return T.Compose([
        T.Resize(int(img_size * 1.14)),   # resize slightly larger (256 for img_size=224)
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
