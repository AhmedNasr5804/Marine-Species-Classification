"""
dataset.py — FathomNet data acquisition, splitting, and DataLoader construction.

Pipeline overview
-----------------
1. load_from_disk()     — scans an existing data/ directory, discovers class
                          folders, and returns concept_files without any API
                          calls. Use this when crops are already downloaded.
2. download_dataset()   — queries FathomNet via the fathomnet-py library and
                          downloads missing crops. Skip if data already exists.
3. build_splits()       — stratified 70/15/15 train/val/test split over known
                          classes; OOD classes kept entirely separate.
                          Splits persisted to splits.json for reproducibility.
4. MarineDataset        — torch Dataset wrapping file paths + integer labels.
5. get_dataloaders()    — returns {train, val, test, ood} DataLoaders.

Class selection rationale
-------------------------
Classes were selected from folders already present in data/ (downloaded in
prior runs). Folder names use underscores for spaces; load_from_disk()
converts them back to the concept name. Only folders with ≥ MIN_PER_CLASS
valid image files are kept. OOD concepts are withheld entirely from training.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .transforms import get_eval_transform, get_train_transform

# ---------------------------------------------------------------------------
# Default concept lists  (verified against live FathomNet, April 2026)
# ---------------------------------------------------------------------------

# 12 known classes — folders verified present and populated in data/ (April 2026).
# Folder name → concept name: underscores replaced with spaces.
KNOWN_CONCEPTS: List[str] = [
    "Actiniaria",              # sea anemones          (150 crops on disk)
    "Crinoidea",               # feather stars         (150 crops on disk)
    "Holothuroidea",           # sea cucumbers         (150 crops on disk)
    "Ophiuroidea",             # brittle stars         (150 crops on disk)
    "Pennatulacea",            # sea pens              (150 crops on disk)
    "Porifera",                # sponges               (150 crops on disk)
    "Sebastolobus",            # thornyhead rockfish   (150 crops on disk)
    "Nanomia",                 # siphonophore genus    (150 crops on disk)
    "Paragorgia arborea",      # bubblegum coral       (140 crops on disk)
    "Dosidicus gigas",         # Humboldt squid        (140 crops on disk)
    "Beroe abyssicola",        # comb jelly            (140 crops on disk)
    "Bathochordaeus stygius",  # giant larvacean       (140 crops on disk)
]

# 3 OOD (unknown) classes — withheld entirely from training and validation.
# Chosen to be taxonomically adjacent but visually distinct from known classes.
OOD_CONCEPTS: List[str] = [
    "Aegina citrea",           # hydromedusa           (140 crops on disk)
    "Pleuroncodes planipes",   # pelagic red crab      (116 crops on disk)
    "Acanthogorgia",           # thorny coral          (74 crops on disk)
]

ALL_CONCEPTS: List[str] = KNOWN_CONCEPTS + OOD_CONCEPTS

# Maximum bounding-box crops to download per concept (class balance cap)
MAX_PER_CLASS: int = 150
# Drop a concept if fewer crops are available after download
MIN_PER_CLASS: int = 20

SPLIT_FRACTIONS: Tuple[float, float, float] = (0.70, 0.15, 0.15)

# ---------------------------------------------------------------------------
# Load existing data from disk (no API calls)
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_from_disk(
    data_dir: str | Path,
    concepts: Optional[List[str]] = None,
    min_per_class: int = MIN_PER_CLASS,
) -> Dict[str, List[Path]]:
    """Scan an existing data directory and return concept_files without downloading.

    Each subdirectory of data_dir is treated as one class. The folder name is
    converted back to a concept name by replacing underscores with spaces.
    Only folders whose concept name appears in *concepts* AND that contain at
    least *min_per_class* valid image files are included.

    Use this function instead of download_dataset() when crops are already on disk.

    Args:
        data_dir:      Root data directory (e.g. Path("data")).
        concepts:      Concept names to include. Defaults to ALL_CONCEPTS.
                       Pass None to auto-discover every populated folder.
        min_per_class: Drop a folder if it has fewer than this many images.

    Returns:
        Dict mapping concept_name → list of image Paths (same format as
        download_dataset()).
    """
    if concepts is None:
        concepts = ALL_CONCEPTS

    data_dir = Path(data_dir)
    # Build a lookup: slug → concept name
    slug_to_concept = {c.replace(" ", "_"): c for c in concepts}

    concept_files: Dict[str, List[Path]] = {}

    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue

        concept = slug_to_concept.get(folder.name)
        if concept is None:
            # Folder exists on disk but not in our concept list — skip silently
            continue

        # Collect valid image files, ignoring Zone.Identifier and other metadata
        images = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
            and ":Zone.Identifier" not in p.name
        ])

        if len(images) < min_per_class:
            print(f"  [skip] '{concept}': only {len(images)} images "
                  f"(min={min_per_class})")
            concept_files[concept] = []
        else:
            concept_files[concept] = images

    # Report
    found    = [c for c, f in concept_files.items() if f]
    missing  = [c for c in concepts if c not in concept_files]
    print(f"\n[load_from_disk] Found {len(found)}/{len(concepts)} concepts on disk.")
    for c in sorted(found):
        tag = "(OOD)" if c in OOD_CONCEPTS else "     "
        print(f"  {tag}  {c:<45s}: {len(concept_files[c]):>4d} images")
    if missing:
        print(f"  [warn] Not found on disk: {missing}")

    return concept_files


# ---------------------------------------------------------------------------
# Image download helper
# ---------------------------------------------------------------------------

def _download_image(url: str, retries: int = 3, backoff: float = 1.5) -> Optional[Image.Image]:
    """Download an image from a URL; return PIL Image or None on failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as exc:
            if attempt == retries - 1:
                return None
            time.sleep(backoff * (attempt + 1))
    return None


def _crop_bbox(img: Image.Image, x: int, y: int, width: int, height: int,
               padding: float = 0.1) -> Image.Image:
    """Crop a bounding box with proportional padding for environmental context."""
    pad_w = int(width  * padding)
    pad_h = int(height * padding)
    left   = max(0, x - pad_w)
    top    = max(0, y - pad_h)
    right  = min(img.width,  x + width  + pad_w)
    bottom = min(img.height, y + height + pad_h)
    return img.crop((left, top, right, bottom))


# ---------------------------------------------------------------------------
# Download entry point
# ---------------------------------------------------------------------------

def download_dataset(
    data_dir: str | Path,
    concepts: Optional[List[str]] = None,
    max_per_class: int = MAX_PER_CLASS,
    min_per_class: int = MIN_PER_CLASS,
    skip_existing: bool = True,
) -> Dict[str, List[Path]]:
    """Download FathomNet bounding-box crops for each concept.

    Uses `fathomnet.api.images.find_by_concept()` which returns images with
    embedded bounding-box annotations — no separate per-image API call needed.

    Crops are saved as::

        data_dir/<concept_slug>/<image_uuid>_<bbox_idx>.jpg

    Args:
        data_dir:       Root directory for downloaded crops.
        concepts:       List of FathomNet concept names. Defaults to ALL_CONCEPTS.
        max_per_class:  Cap on crops per concept (class balance).
        min_per_class:  Drop concept if fewer crops obtained.
        skip_existing:  Skip downloading crops already on disk.

    Returns:
        Mapping from concept name → list of crop Paths.
    """
    # Import here so the module is importable even without fathomnet installed
    try:
        from fathomnet.api import images as fm_images
    except ImportError:
        raise ImportError(
            "fathomnet-py is required. Install with: pip install fathomnet"
        )

    if concepts is None:
        concepts = ALL_CONCEPTS

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    concept_files: Dict[str, List[Path]] = {}

    for concept in concepts:
        slug      = concept.replace(" ", "_")
        class_dir = data_dir / slug
        class_dir.mkdir(exist_ok=True)

        print(f"\n[{concept}] Querying FathomNet …")
        try:
            image_dtos = fm_images.find_by_concept(concept)
        except Exception as exc:
            print(f"  [warn] API error for '{concept}': {exc}")
            concept_files[concept] = []
            continue

        if not image_dtos:
            print(f"  [skip] No images found for '{concept}'.")
            concept_files[concept] = []
            continue

        print(f"  Found {len(image_dtos)} images. Extracting up to {max_per_class} crops …")

        saved: List[Path] = []

        for img_dto in tqdm(image_dtos, desc=f"  {slug}", leave=False):
            if len(saved) >= max_per_class:
                break

            if not img_dto.url or not img_dto.boundingBoxes:
                continue

            # Filter bounding boxes belonging to this concept
            bboxes = [
                bb for bb in img_dto.boundingBoxes
                if bb.concept == concept
                   and bb.width and bb.height
                   and bb.width > 0 and bb.height > 0
            ]
            if not bboxes:
                continue

            # Download source image once per image_dto
            img_pil = None

            for idx, bb in enumerate(bboxes):
                if len(saved) >= max_per_class:
                    break

                # Derive a stable filename from the image URL and bbox index
                url_stem  = Path(img_dto.url).stem
                crop_path = class_dir / f"{url_stem}_{idx}.jpg"

                if skip_existing and crop_path.exists():
                    saved.append(crop_path)
                    continue

                if img_pil is None:
                    img_pil = _download_image(img_dto.url)
                    if img_pil is None:
                        break   # skip all bboxes for this image

                x, y, w, h = bb.x or 0, bb.y or 0, bb.width, bb.height
                crop = _crop_bbox(img_pil, x, y, w, h)

                if crop.width < 32 or crop.height < 32:
                    continue  # discard degenerate crops

                crop.save(crop_path, "JPEG", quality=90)
                saved.append(crop_path)

        if len(saved) < min_per_class:
            print(f"  [warn] Only {len(saved)} crops for '{concept}' "
                  f"(min={min_per_class}). Class excluded.")
            concept_files[concept] = []
        else:
            print(f"  Saved {len(saved)} crops.")
            concept_files[concept] = saved

    return concept_files


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def build_splits(
    concept_files: Dict[str, List[Path]],
    ood_concepts: Optional[List[str]] = None,
    fractions: Tuple[float, float, float] = SPLIT_FRACTIONS,
    split_file: Optional[str | Path] = None,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, int]]]:
    """Create stratified train/val/test/ood splits and persist to JSON.

    OOD concepts are placed entirely in the "ood" split (label = -1) and
    are never exposed to training or validation.

    Args:
        concept_files: Output of download_dataset().
        ood_concepts:  Concept names to route to OOD split.
        fractions:     (train, val, test) proportions for known classes.
        split_file:    Path to save/load the JSON split file.
        seed:          RNG seed for the stratified shuffle.

    Returns:
        Dict with keys "train", "val", "test", "ood"; each a list of
        (file_path_str, label_int) tuples.
    """
    import random as _random

    if ood_concepts is None:
        ood_concepts = OOD_CONCEPTS

    split_file = Path(split_file) if split_file else None

    # Load from cache if available
    if split_file and split_file.exists():
        print(f"[splits] Loading existing splits from {split_file}")
        with open(split_file) as f:
            raw = json.load(f)
        return {k: [(p, i) for p, i in v] for k, v in raw.items()}

    ood_set = set(ood_concepts)

    # Build ordered label map for known classes
    known_concepts = sorted([
        c for c, files in concept_files.items()
        if c not in ood_set and files
    ])
    label_map: Dict[str, int] = {c: i for i, c in enumerate(known_concepts)}

    rng = _random.Random(seed)
    splits: Dict[str, List[Tuple[str, int]]] = {
        "train": [], "val": [], "test": [], "ood": []
    }

    train_f, val_f, _ = fractions
    for concept, files in concept_files.items():
        if not files:
            continue
        paths = [str(p) for p in files]
        rng.shuffle(paths)

        if concept in ood_set:
            splits["ood"].extend((p, -1) for p in paths)
        elif concept in label_map:
            label   = label_map[concept]
            n       = len(paths)
            n_train = max(1, int(n * train_f))
            n_val   = max(1, int(n * val_f))
            splits["train"].extend((p, label) for p in paths[:n_train])
            splits["val"].extend(  (p, label) for p in paths[n_train:n_train + n_val])
            splits["test"].extend( (p, label) for p in paths[n_train + n_val:])

    if split_file:
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"[splits] Saved to {split_file}")

    _print_split_summary(splits, label_map)
    return splits


def _print_split_summary(
    splits: Dict[str, List[Tuple[str, int]]],
    label_map: Dict[str, int],
) -> None:
    total = sum(len(v) for v in splits.values())
    print("\n=== Split summary ===")
    for name, items in splits.items():
        print(f"  {name:5s}: {len(items):5d} samples")
    print(f"  {'total':5s}: {total:5d} samples")
    print(f"  Known classes ({len(label_map)}): {list(label_map.keys())}")


def get_class_names(concept_files: Dict[str, List[Path]],
                    ood_concepts: Optional[List[str]] = None) -> List[str]:
    """Return ordered list of known class names (matches integer labels)."""
    if ood_concepts is None:
        ood_concepts = OOD_CONCEPTS
    ood_set = set(ood_concepts)
    return sorted([c for c, files in concept_files.items()
                   if c not in ood_set and files])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MarineDataset(Dataset):
    """Torch Dataset for marine species classification crops."""

    def __init__(self, samples: List[Tuple[str, int]],
                 transform: Optional[T.Compose] = None) -> None:
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    splits: Dict[str, List[Tuple[str, int]]],
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Build DataLoaders for all four splits.

    Args:
        splits:      Output of build_splits().
        batch_size:  Mini-batch size (hyperparameter).
        num_workers: Parallel data-loading workers.
        img_size:    Spatial resolution for transforms.
        pin_memory:  Enable for GPU training (faster host→device transfers).

    Returns:
        Dict with keys "train", "val", "test", "ood".
    """
    train_tf = get_train_transform(img_size)
    eval_tf  = get_eval_transform(img_size)

    loaders: Dict[str, DataLoader] = {}
    for split_name, samples in splits.items():
        if not samples:
            continue
        tf      = train_tf if split_name == "train" else eval_tf
        ds      = MarineDataset(samples, transform=tf)
        shuffle = (split_name == "train")
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_name == "train"),
        )
    return loaders
