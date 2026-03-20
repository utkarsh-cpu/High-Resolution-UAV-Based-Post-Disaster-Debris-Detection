"""
Unit tests for the dataset download pipeline.
Run with:  pytest hurricane_debris/tests/test_download.py -v

These tests use importlib to load ``download.py`` directly so they run in
a lightweight environment (no torch / cv2 required).
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import types
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Load download.py without triggering the heavy package __init__.py ─────────

def _load_module_direct(rel_path: str, module_name: str):
    """Import a module from a file path, bypassing package __init__ chains."""
    src = Path(__file__).parent.parent.parent / rel_path
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)

    # hurricane_debris.utils.logging must be importable; stub it out so we
    # don't drag in the full package either.
    if "hurricane_debris.utils.logging" not in sys.modules:
        import logging
        _log_mod = types.ModuleType("hurricane_debris.utils.logging")
        _log_mod.get_logger = lambda name: logging.getLogger(name)  # type: ignore[attr-defined]
        sys.modules["hurricane_debris.utils.logging"] = _log_mod
    if "hurricane_debris.utils" not in sys.modules:
        _utils_mod = types.ModuleType("hurricane_debris.utils")
        sys.modules["hurricane_debris.utils"] = _utils_mod
    if "hurricane_debris" not in sys.modules:
        _pkg_mod = types.ModuleType("hurricane_debris")
        sys.modules["hurricane_debris"] = _pkg_mod

    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_dl = _load_module_direct(
    "hurricane_debris/data/download.py",
    "hurricane_debris.data.download",
)

DATASET_REGISTRY = _dl.DATASET_REGISTRY
DatasetInfo = _dl.DatasetInfo
_extract_archive = _dl._extract_archive
_flatten_single_subdir = _dl._flatten_single_subdir
_validate_dataset_dir = _dl._validate_dataset_dir
download_dataset = _dl.download_dataset
verify_dataset = _dl.verify_dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a fresh temporary directory for each test."""
    return tmp_path


def _make_zip(dest: Path, inner_files: dict[str, str]) -> Path:
    """Create a zip archive at *dest* containing *inner_files* (name → content)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w") as zf:
        for name, content in inner_files.items():
            zf.writestr(name, content)
    return dest


# ── Registry tests ────────────────────────────────────────────────────────────


class TestDatasetRegistry:
    def test_all_expected_datasets_registered(self):
        for name in ("rescuenet", "msnet", "designsafe"):
            assert name in DATASET_REGISTRY

    def test_dataset_info_has_required_fields(self):
        for name, info in DATASET_REGISTRY.items():
            assert isinstance(info, DatasetInfo)
            assert info.name == name
            assert info.description
            assert info.archive_name
            assert isinstance(info.expected_dirs, list)
            assert isinstance(info.manual_instructions, str)

    def test_manual_instructions_are_non_empty(self):
        for info in DATASET_REGISTRY.values():
            assert len(info.manual_instructions) > 10, (
                f"{info.name}: manual_instructions should not be empty"
            )

    def test_rescuenet_manual_instructions_cover_colourmask_dropbox_layout(self):
        instructions = DATASET_REGISTRY["rescuenet"].manual_instructions
        assert "ColorMasks-RescueNet" in instructions
        assert "RescueNet/" in instructions


# ── Validation helpers ────────────────────────────────────────────────────────


class TestValidateDatasetDir:
    def test_returns_true_when_all_dirs_exist(self, tmp_dir):
        (tmp_dir / "train").mkdir()
        (tmp_dir / "val").mkdir()
        assert _validate_dataset_dir(tmp_dir, ["train", "val"]) is True

    def test_returns_false_when_dir_missing(self, tmp_dir):
        (tmp_dir / "train").mkdir()
        assert _validate_dataset_dir(tmp_dir, ["train", "val"]) is False

    def test_returns_true_for_empty_expected(self, tmp_dir):
        assert _validate_dataset_dir(tmp_dir, []) is True


# ── Extraction tests ──────────────────────────────────────────────────────────


class TestExtractArchive:
    def test_extracts_zip(self, tmp_dir):
        archive = _make_zip(
            tmp_dir / "test.zip",
            {"images/a.png": "fake-image", "annotations/b.json": "{}"},
        )
        dest = tmp_dir / "out"
        _extract_archive(archive, dest)
        assert (dest / "images" / "a.png").exists()
        assert (dest / "annotations" / "b.json").exists()

    def test_raises_on_unsupported_format(self, tmp_dir):
        fake = tmp_dir / "data.rar"
        fake.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported archive format"):
            _extract_archive(fake, tmp_dir / "out")


# ── Flatten helper ────────────────────────────────────────────────────────────


class TestFlattenSingleSubdir:
    def test_flattens_single_child_directory(self, tmp_dir):
        # Simulate extraction that produces a single top-level dir
        sub = tmp_dir / "rescuenet_v1"
        sub.mkdir()
        (sub / "train").mkdir()
        (sub / "train" / "image.png").write_text("x")

        _flatten_single_subdir(tmp_dir)

        assert (tmp_dir / "train").exists()
        assert (tmp_dir / "train" / "image.png").exists()
        # The intermediate directory should be gone
        assert not sub.exists()

    def test_does_not_flatten_multiple_children(self, tmp_dir):
        (tmp_dir / "dir_a").mkdir()
        (tmp_dir / "dir_b").mkdir()
        # Should leave layout unchanged
        _flatten_single_subdir(tmp_dir)
        assert (tmp_dir / "dir_a").exists()
        assert (tmp_dir / "dir_b").exists()


# ── download_dataset ──────────────────────────────────────────────────────────


class TestDownloadDataset:
    def test_raises_on_unknown_dataset(self, tmp_dir):
        with pytest.raises(ValueError, match="Unknown dataset"):
            download_dataset("nonexistent", dest_dir=str(tmp_dir))

    def test_skips_download_when_data_exists(self, tmp_dir):
        """If expected_dirs already exist, skip download without error."""
        info = DATASET_REGISTRY["rescuenet"]
        ds_dir = tmp_dir / "rescuenet"
        for sub in info.expected_dirs:
            (ds_dir / sub).mkdir(parents=True)

        # No actual download should happen – gdown / URL should not be called
        result = download_dataset("rescuenet", dest_dir=str(tmp_dir))
        assert result == ds_dir

    def test_skips_download_for_dropbox_rescuenet_colourmask_layout(self, tmp_dir):
        info = DATASET_REGISTRY["rescuenet"]
        image_root = tmp_dir / "RescueNet"
        mask_root = tmp_dir / "ColorMasks-RescueNet"
        for sub in info.expected_dirs:
            (image_root / sub).mkdir(parents=True)
            (mask_root / sub).mkdir(parents=True)

        result = download_dataset("rescuenet", dest_dir=str(tmp_dir))
        assert result == image_root

    def test_force_retriggers_download_attempt(self, tmp_dir):
        """With force=True, download is attempted even if data exists."""
        info = DATASET_REGISTRY["rescuenet"]
        ds_dir = tmp_dir / "rescuenet"
        for sub in info.expected_dirs:
            (ds_dir / sub).mkdir(parents=True)

        # _try_download will be called; since no real network is available it
        # should fail → RuntimeError
        with patch.object(_dl, "_try_download", return_value=False):
            with pytest.raises(RuntimeError, match="Automatic download"):
                download_dataset("rescuenet", dest_dir=str(tmp_dir), force=True)

    def test_raises_when_no_download_source(self, tmp_dir):
        """RuntimeError is raised when both gdrive_id and url are None."""
        with patch.object(_dl, "_try_download", return_value=False):
            with pytest.raises(RuntimeError, match="Automatic download"):
                download_dataset("msnet", dest_dir=str(tmp_dir))

    def test_successful_gdown_download_and_extraction(self, tmp_dir):
        """Simulate a successful gdown download with a real zip archive."""
        # Create a fake archive in a separate dir (different from dest_dir)
        src_archive = tmp_dir / "src" / "rescuenet.zip"
        _make_zip(
            src_archive,
            {
                "train/.keep": "",
                "val/.keep": "",
                "test/.keep": "",
            },
        )
        dest_dir = tmp_dir / "datasets"

        def fake_try_download(info, dest_path):
            shutil.copy(str(src_archive), str(dest_path))
            return True

        with patch.object(_dl, "_try_download", side_effect=fake_try_download):
            result = download_dataset("rescuenet", dest_dir=str(dest_dir))

        assert result.exists()
        for sub in DATASET_REGISTRY["rescuenet"].expected_dirs:
            assert (result / sub).exists()

    def test_keep_archive_flag(self, tmp_dir):
        """The archive should persist when keep_archive=True."""
        src_archive = tmp_dir / "src" / "rescuenet.zip"
        _make_zip(src_archive, {"train/.keep": "", "val/.keep": "", "test/.keep": ""})
        dest_dir = tmp_dir / "datasets"

        def fake_try_download(info, dest_path):
            shutil.copy(str(src_archive), str(dest_path))
            return True

        with patch.object(_dl, "_try_download", side_effect=fake_try_download):
            download_dataset(
                "rescuenet", dest_dir=str(dest_dir), keep_archive=True
            )

        assert (dest_dir / "rescuenet.zip").exists()

    def test_download_all_dispatches_to_each_dataset(self, tmp_dir):
        """--dataset all should invoke download for every registered dataset."""
        # Pre-create expected dirs to satisfy the "already exists" check
        # for every dataset, so no actual download is triggered.
        for name, info in DATASET_REGISTRY.items():
            ds_dir = tmp_dir / name
            for sub in info.expected_dirs:
                (ds_dir / sub).mkdir(parents=True, exist_ok=True)

        result = download_dataset("all", dest_dir=str(tmp_dir))
        assert result == tmp_dir


# ── verify_dataset ────────────────────────────────────────────────────────────


class TestVerifyDataset:
    def test_returns_true_for_valid_layout(self, tmp_dir):
        info = DATASET_REGISTRY["rescuenet"]
        ds_dir = tmp_dir / "rescuenet"
        for sub in info.expected_dirs:
            (ds_dir / sub).mkdir(parents=True)

        assert verify_dataset("rescuenet", str(ds_dir)) is True

    def test_returns_false_for_incomplete_layout(self, tmp_dir):
        ds_dir = tmp_dir / "rescuenet"
        ds_dir.mkdir()
        assert verify_dataset("rescuenet", str(ds_dir)) is False

    def test_returns_true_for_dropbox_rescuenet_colourmask_layout(self, tmp_dir):
        info = DATASET_REGISTRY["rescuenet"]
        for sub in info.expected_dirs:
            (tmp_dir / "RescueNet" / sub).mkdir(parents=True)
            (tmp_dir / "ColorMasks-RescueNet" / sub).mkdir(parents=True)

        assert verify_dataset("rescuenet", str(tmp_dir)) is True

    def test_raises_on_unknown_dataset(self, tmp_dir):
        with pytest.raises(ValueError, match="Unknown dataset"):
            verify_dataset("unknown_ds", str(tmp_dir))
