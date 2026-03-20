"""
Dataset Download Pipeline
=========================
Downloads and prepares the three datasets used in this project:

  - **RescueNet** (Alam et al., IEEE TGRS 2022) – semantic segmentation of
    Hurricane-Michael imagery.  Hosted on Google Drive via ``gdown``.
  - **MSNet** (Xie & Xiong, Remote Sensing 2023) – multi-scale damage
    classification.  COCO-format annotations hosted on a direct HTTP URL.
  - **DesignSafe-CI PRJ-6029** (NSF NHERI) – multi-hazard UAV imagery.
    Requires manual download from the DesignSafe portal.

Usage via CLI (recommended):
    python main.py --download --dataset rescuenet --dataset-dir ./datasets
    python main.py --download --dataset msnet     --dataset-dir ./datasets
    python main.py --download --dataset all       --dataset-dir ./datasets

Usage as library:
    from hurricane_debris.data.download import download_dataset
    download_dataset("rescuenet", dest_dir="./datasets")

Each download function:
  1. Skips gracefully if the destination already looks valid.
  2. Downloads the archive (zip / tar.gz) with a progress bar.
  3. Extracts it into the expected directory layout.
  4. Validates the structure and prints a summary.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlretrieve

from hurricane_debris.utils.logging import get_logger

logger = get_logger("data.download")
_LAYOUT_DEFAULT = "default"
_LAYOUT_RESCUENET = "rescuenet"

# ── Dataset registry ──────────────────────────────────────────────────────────

@dataclass
class DatasetInfo:
    """Metadata for a downloadable dataset."""

    name: str
    description: str
    #: gdrive_id  → downloaded via gdown (Google Drive); or
    #: url        → downloaded via urllib / requests (direct HTTP)
    gdrive_id: Optional[str] = None
    url: Optional[str] = None
    archive_name: str = "archive.zip"
    #: Alternate single local archive names accepted for manual placement.
    #: These are one-file alternatives checked after `archive_name`.
    #: Example: MSNet accepts either `msnet.zip` or `ISBDA.zip`.
    local_archive_names: List[str] = field(default_factory=list)
    #: A complete set of local archives that must all be present together.
    #: Unlike `local_archive_names`, every archive in this bundle must exist.
    #: Example: RescueNet uses `RescueNet.zip` plus `ColorMasks-RescueNet.zip`.
    local_archive_bundle: List[str] = field(default_factory=list)
    #: When True, extract a found archive bundle into the dataset root folder.
    extract_bundle_into_root: bool = False
    #: Dataset layout resolver key used for pre/post-extraction validation.
    existing_dir_layout: str = _LAYOUT_DEFAULT
    #: Expected top-level subdirectories after extraction
    expected_dirs: List[str] = field(default_factory=list)
    #: Human-readable instructions when automated download is unavailable
    manual_instructions: str = ""


# Public dataset registry.  Fill in gdrive_id / url when official mirrors
# are available.  The manual_instructions field is always shown when the
# automated path fails so users can obtain the data themselves.
DATASET_REGISTRY: Dict[str, DatasetInfo] = {
    "rescuenet": DatasetInfo(
        name="rescuenet",
        description=(
            "RescueNet: 4,494 post-Hurricane-Michael UAV images with "
            "pixel-level semantic segmentation (8 classes, 0.5–2 cm/px GSD)."
        ),
        # Public Google Drive mirror maintained by the RescueNet authors.
        # Set to None if the link expires; supply --url or use manual download.
        gdrive_id="1alu8k9WuYFoMizuBxGpLYsS5S9rCUHJP",
        archive_name="rescuenet.zip",
        local_archive_bundle=["RescueNet.zip", "ColorMasks-RescueNet.zip"],
        extract_bundle_into_root=True,
        existing_dir_layout=_LAYOUT_RESCUENET,
        expected_dirs=["train", "val", "test"],
        manual_instructions=(
            "1. Register at https://ieee-dataport.org/open-access/rescuenet\n"
            "2. Download the RescueNet imagery plus the optional "
            "ColorMasks-RescueNet / colormask-rescuenet archive from the official mirrors.\n"
            "3. Either extract into  <dataset-dir>/rescuenet/  so that the layout is:\n"
            "     rescuenet/\n"
            "       train/train-org-img/\n"
            "       train/train-label-img/\n"
            "       val/val-org-img/\n"
            "       val/val-label-img/\n"
            "       test/test-org-img/\n"
            "       test/test-label-img/\n"
            "   or keep the Dropbox layout as sibling folders:\n"
            "     <dataset-dir>/RescueNet/\n"
            "     <dataset-dir>/ColorMasks-RescueNet/\n"
            "   The RescueNet loader supports both layouts."
        ),
    ),
    "msnet": DatasetInfo(
        name="msnet",
        description=(
            "MSNet: 8,700+ multi-scale disaster images across 7 events with "
            "COCO-format instance annotations."
        ),
        gdrive_id=None,
        url=None,
        archive_name="msnet.zip",
        local_archive_names=["ISBDA.zip"],
        expected_dirs=["images", "annotations"],
        manual_instructions=(
            "1. Visit the xBD dataset page or the MSNet repository:\n"
            "   https://github.com/DIUx-xView/xView2_baseline\n"
            "2. Request access and download the dataset.\n"
            "3. Convert annotations to COCO format with:\n"
            "   python scripts/convert_msnet_annotations.py\n"
            "4. Place data under  <dataset-dir>/msnet/  so the layout is:\n"
            "     msnet/\n"
            "       images/\n"
            "       annotations/instances_train.json\n"
            "       annotations/instances_val.json\n"
            "       annotations/instances_test.json"
        ),
    ),
    "designsafe": DatasetInfo(
        name="designsafe",
        description=(
            "DesignSafe-CI PRJ-6029: NSF NHERI multi-hazard UAV and "
            "ground-based imagery (DOI: 10.17603/ds2-jvps-2n95)."
        ),
        gdrive_id=None,
        url=None,
        archive_name="designsafe.zip",
        local_archive_names=["PRJ-6029.zip"],
        expected_dirs=["images", "annotations"],
        manual_instructions=(
            "The DesignSafe-CI dataset requires a free account.\n"
            "1. Register at https://www.designsafe-ci.org/\n"
            "2. Navigate to PRJ-6029 (DOI: 10.17603/ds2-jvps-2n95).\n"
            "3. Download the imagery and damage-observations JSON.\n"
            "4. Place data under  <dataset-dir>/designsafe/  so the layout is:\n"
            "     designsafe/\n"
            "       images/\n"
            "       annotations/damage_observations.json"
        ),
    ),
}
_RESCUENET_COLOURMASK_ROOTS = (
    "ColorMasks-RescueNet",
    "ColourMasks-RescueNet",
    "colormasks-rescuenet",
    "colourmasks-rescuenet",
    "colormask-rescuenet",
    "colourmask-rescuenet",
)

# ── Progress hook for urllib.urlretrieve ─────────────────────────────────────

def _make_reporthook(desc: str = "Downloading"):
    """Return a urlretrieve reporthook that prints download progress."""
    try:
        from tqdm import tqdm  # type: ignore

        pbar: Optional[tqdm] = None

        def hook(block_num: int, block_size: int, total_size: int):
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=desc,
                    leave=True,
                )
            downloaded = block_num * block_size
            if total_size > 0:
                pbar.n = min(downloaded, total_size)
                pbar.refresh()
            if total_size > 0 and downloaded >= total_size:
                pbar.close()

        return hook

    except ImportError:
        # tqdm not available – simple text output
        def hook(block_num: int, block_size: int, total_size: int):  # type: ignore[misc]
            downloaded = block_num * block_size
            if total_size > 0 and block_num % 500 == 0:
                pct = min(100, downloaded * 100 / total_size)
                print(f"\r{desc}: {pct:.1f}%", end="", flush=True)

        return hook


# ── Archive extraction ────────────────────────────────────────────────────────

def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .zip or .tar.gz archive into *dest_dir*."""
    name = archive_path.name.lower()
    logger.info("Extracting %s → %s", archive_path.name, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    logger.info("Extraction complete.")


# ── Integrity check ───────────────────────────────────────────────────────────

def _validate_dataset_dir(dest_dir: Path, expected_dirs: List[str]) -> bool:
    """Return True if all expected top-level subdirectories exist."""
    for sub in expected_dirs:
        if not (dest_dir / sub).exists():
            return False
    return True


def _first_existing_child(base_dir: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return None


def _resolve_existing_rescuenet_dir(dest_root: Path, dataset_dir: Path) -> Optional[Path]:
    """Return an existing RescueNet root for flat or Dropbox layouts."""
    if _validate_dataset_dir(dataset_dir, DATASET_REGISTRY["rescuenet"].expected_dirs):
        return dataset_dir

    if _validate_dataset_dir(dest_root / "RescueNet", DATASET_REGISTRY["rescuenet"].expected_dirs):
        colourmask_root = _first_existing_child(dest_root, list(_RESCUENET_COLOURMASK_ROOTS))
        if colourmask_root is not None and _validate_dataset_dir(
            colourmask_root, DATASET_REGISTRY["rescuenet"].expected_dirs
        ):
            return dest_root / "RescueNet"

    if _validate_dataset_dir(dataset_dir / "RescueNet", DATASET_REGISTRY["rescuenet"].expected_dirs):
        colourmask_root = _first_existing_child(dataset_dir, list(_RESCUENET_COLOURMASK_ROOTS))
        if colourmask_root is not None and _validate_dataset_dir(
            colourmask_root, DATASET_REGISTRY["rescuenet"].expected_dirs
        ):
            return dataset_dir / "RescueNet"

    return None


# ── Per-dataset download implementations ─────────────────────────────────────

def _download_via_gdown(gdrive_id: str, dest_path: Path) -> bool:
    """Download a file from Google Drive using gdown.

    Returns True on success, False if gdown is unavailable or download fails.
    """
    try:
        import gdown  # type: ignore
    except ImportError:
        logger.warning(
            "gdown is not installed.  Install it with: pip install gdown>=4.7.1"
        )
        return False

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    logger.info("Downloading from Google Drive (id=%s) …", gdrive_id)
    try:
        gdown.download(url, str(dest_path), quiet=False, fuzzy=True)
        return dest_path.exists() and dest_path.stat().st_size > 0
    except Exception as exc:
        logger.error("gdown download failed: %s", exc)
        return False


def _download_via_url(url: str, dest_path: Path) -> bool:
    """Download a file from a direct HTTPS URL.

    Only ``https://`` URLs are accepted to prevent accidental use of
    insecure or unexpected protocols.

    Returns True on success, False on failure.
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("https",):
        logger.error(
            "Refusing to download from non-HTTPS URL (scheme=%r): %s",
            parsed.scheme, url,
        )
        return False

    logger.info("Downloading from %s …", url)
    try:
        hook = _make_reporthook(f"Downloading {dest_path.name}")
        urlretrieve(url, str(dest_path), reporthook=hook)  # noqa: S310
        print()  # newline after progress
        return dest_path.exists() and dest_path.stat().st_size > 0
    except Exception as exc:
        logger.error("HTTP download failed: %s", exc)
        return False


def _try_download(info: DatasetInfo, archive_path: Path) -> bool:
    """Attempt automatic download; return True on success."""
    if info.gdrive_id:
        if _download_via_gdown(info.gdrive_id, archive_path):
            return True
        logger.warning("Google Drive download failed; trying direct URL …")

    if info.url:
        if _download_via_url(info.url, archive_path):
            return True
        logger.warning("Direct URL download also failed.")

    return False


def _has_existing_archive(archive_path: Path) -> bool:
    """Return True when an archive already exists locally."""
    if archive_path.is_file():
        logger.info("Found existing archive at %s", archive_path)
        return True
    return False


def _find_local_archives(info: DatasetInfo, dest_root: Path) -> List[Path]:
    """Return matching local archive(s) for a dataset, if present."""
    if info.local_archive_bundle:
        split_archives = [dest_root / archive_name for archive_name in info.local_archive_bundle]
        if all(path.is_file() for path in split_archives):
            logger.info(
                "Found complete local archive bundle for '%s'; preferring bundle extraction.",
                info.name,
            )
            for path in split_archives:
                logger.info("Found existing archive at %s", path)
            return split_archives

    for archive_name in [info.archive_name, *info.local_archive_names]:
        archive_path = dest_root / archive_name
        if _has_existing_archive(archive_path):
            return [archive_path]

    return []


def _resolve_existing_dataset_dir(
    info: DatasetInfo,
    dest_root: Path,
    dataset_dir: Path,
) -> Optional[Path]:
    """Return an existing dataset directory for the configured layout."""
    if info.existing_dir_layout == _LAYOUT_RESCUENET:
        return _resolve_existing_rescuenet_dir(dest_root, dataset_dir)
    if _validate_dataset_dir(dataset_dir, info.expected_dirs):
        return dataset_dir
    return None


def _uses_local_archive_bundle(info: DatasetInfo, local_archives: List[Path]) -> bool:
    """Return True when *local_archives* matches the configured archive bundle."""
    if not info.local_archive_bundle:
        return False
    archive_names = {path.name for path in local_archives}
    return archive_names == set(info.local_archive_bundle)


# ── Public API ────────────────────────────────────────────────────────────────

def download_dataset(
    name: str,
    dest_dir: str = "./datasets",
    force: bool = False,
    keep_archive: bool = False,
) -> Path:
    """Download and prepare a named dataset.

    Args:
        name:         Dataset name: ``"rescuenet"``, ``"msnet"``,
                      ``"designsafe"``, or ``"all"``.
        dest_dir:     Root datasets directory.  The dataset will be placed in
                      ``<dest_dir>/<name>/``.
        force:        Re-download even if the directory already looks valid.
        keep_archive: Keep the downloaded archive after extraction.

    Returns:
        Path to the dataset directory (``<dest_dir>/<name>``).

    Raises:
        ValueError: if *name* is not in the registry.
        RuntimeError: if the dataset requires manual download.
    """
    if name == "all":
        failed: List[str] = []
        for ds_name in DATASET_REGISTRY:
            try:
                download_dataset(ds_name, dest_dir=dest_dir, force=force, keep_archive=keep_archive)
            except RuntimeError as exc:
                logger.warning(
                    "Skipping '%s' after error (continuing with remaining datasets): %s",
                    ds_name, exc,
                )
                failed.append(ds_name)
        if failed:
            logger.warning(
                "The following datasets could not be downloaded automatically: %s. "
                "See the instructions printed above for each.",
                failed,
            )
        return Path(dest_dir)

    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}"
        )

    info = DATASET_REGISTRY[name]
    dest_root = Path(dest_dir)
    dataset_dir = dest_root / name

    # ── Already exists? ──────────────────────────────────────────────────
    if not force:
        existing_dataset_dir = _resolve_existing_dataset_dir(info, dest_root, dataset_dir)
        if existing_dataset_dir is not None:
            logger.info(
                "Dataset '%s' already present at %s – skipping download.",
                name, existing_dataset_dir,
            )
            return existing_dataset_dir

    logger.info("=" * 60)
    logger.info("Dataset: %s", info.name.upper())
    logger.info("%s", info.description)
    logger.info("Destination: %s", dataset_dir)
    logger.info("=" * 60)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ─────────────────────────────────────────────────────────
    archive_path = dest_root / info.archive_name
    local_archives = _find_local_archives(info, dest_root)
    if local_archives:
        logger.info(
            "Using existing local archive(s) for '%s' instead of downloading again.",
            name,
        )
        archive_ready = True
        extract_root = (
            dest_root
            if info.extract_bundle_into_root and _uses_local_archive_bundle(info, local_archives)
            else dataset_dir
        )
    else:
        archive_ready = _try_download(info, archive_path)
        local_archives = [archive_path]
        extract_root = dataset_dir

    if not archive_ready:
        _print_manual_instructions(info)
        raise RuntimeError(
            f"Automatic download for '{name}' is unavailable. "
            "See instructions above to obtain the data manually."
        )

    # ── Extract ──────────────────────────────────────────────────────────
    for current_archive in local_archives:
        _extract_archive(current_archive, extract_root)

    # After extraction the archive may have created a single top-level
    # directory.  Flatten it if needed so that expected_dirs are direct
    # children of dataset_dir.
    if extract_root == dataset_dir:
        _flatten_single_subdir(dataset_dir)
    # Bundles extracted into dest_root intentionally keep their sibling layout
    # (for example RescueNet/ plus ColorMasks-RescueNet/).

    if not keep_archive:
        for current_archive in local_archives:
            current_archive.unlink(missing_ok=True)
            logger.info("Archive removed: %s", current_archive)

    # ── Validate ─────────────────────────────────────────────────────────
    resolved_dataset_dir = _resolve_existing_dataset_dir(info, dest_root, dataset_dir)

    if resolved_dataset_dir is not None:
        logger.info(
            "✓ Dataset '%s' ready at %s", name, resolved_dataset_dir
        )
    else:
        missing = [
            sub for sub in info.expected_dirs
            if not (dataset_dir / sub).exists()
        ]
        logger.warning(
            "Dataset '%s' extracted but some expected directories are missing: %s. "
            "The dataset may still be usable if the layout differs slightly.",
            name,
            missing,
        )

    # Fall back to dataset_dir for callers that want to inspect a partially
    # extracted dataset after the warning above.
    return resolved_dataset_dir or dataset_dir


def _flatten_single_subdir(dest_dir: Path) -> None:
    """If extraction created exactly one subdirectory, move its contents up."""
    children = [p for p in dest_dir.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        sub = children[0]
        # Move all contents of sub/ into dest_dir/
        for item in list(sub.iterdir()):
            target = dest_dir / item.name
            if target.exists():
                logger.warning(
                    "Skipping move of '%s': target '%s' already exists.",
                    item.name, target,
                )
                continue
            shutil.move(str(item), str(target))
        # Remove the (now empty) subdirectory
        try:
            sub.rmdir()
        except OSError:
            pass  # Not empty – leave it; layout may still be usable


def _print_manual_instructions(info: DatasetInfo) -> None:
    """Print formatted manual download instructions to the logger."""
    border = "=" * 60
    logger.info(border)
    logger.info("MANUAL DOWNLOAD REQUIRED: %s", info.name.upper())
    logger.info(border)
    for line in info.manual_instructions.splitlines():
        logger.info("  %s", line)
    logger.info(border)


def verify_dataset(name: str, dataset_dir: str) -> bool:
    """Check whether a dataset directory has the expected structure.

    Args:
        name:        Dataset name (``"rescuenet"``, ``"msnet"``, ``"designsafe"``).
        dataset_dir: Path to the dataset root (``<datasets>/<name>``).

    Returns:
        True if the directory looks valid, False otherwise.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'")
    info = DATASET_REGISTRY[name]
    path = Path(dataset_dir)
    if name == "rescuenet":
        valid_root = _resolve_existing_rescuenet_dir(path.parent, path)
        if valid_root is None:
            valid_root = _resolve_existing_rescuenet_dir(path, path / name)
        valid = valid_root is not None
    else:
        valid = _validate_dataset_dir(path, info.expected_dirs)
    if valid:
        logger.info("✓ %s: directory structure looks valid at %s", name, path)
    else:
        missing = [d for d in info.expected_dirs if not (path / d).exists()]
        logger.warning(
            "✗ %s: missing expected sub-directories %s under %s",
            name, missing, path,
        )
    return valid
