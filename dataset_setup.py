from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from utils.matlab_resize import imresize


SWINIR_BENCHMARK_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u"
)
DATASET_ALIASES = {
    "Set5": ("Set5", "set5"),
    "Set14": ("Set14", "set14"),
    "BSD100": ("BSD100", "BSDS100", "B100", "bsd100", "bsds100", "b100"),
    "Urban100": ("Urban100", "urban100"),
}
DATASET_ARCHIVES = {
    "Set5": ("Set5.zip", "Set5", "1DnHLNkcpl0wLznwAGW6CcrMMJOZY8ILz"),
    "Set14": ("Set14.zip", "Set14", "1YC6l1o8qBtkU4LUtBQbOZ5sIM-lZf7YO"),
    "BSD100": ("BSDS100.zip", "BSDS100", "1-Qr2vcE8iXfTta0pm9uvuLrRD84s7Rxd"),
    "Urban100": ("urban100.zip", "urban100", "1UlNulSoyflrEObwu19BBlT7f2_Wxycga"),
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def ensure_benchmark_datasets(
    dataset_root: str,
    datasets: Iterable[str],
    scale_factor: int,
    auto_download: bool = True,
) -> None:
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    missing = _missing_datasets(dataset_root_path, datasets, scale_factor)
    if not missing:
        return

    _prepare_from_existing_sources(dataset_root_path, missing, scale_factor)
    missing = _missing_datasets(dataset_root_path, datasets, scale_factor)
    if not missing:
        return

    if not auto_download:
        raise FileNotFoundError(
            "Missing benchmark datasets: "
            + ", ".join(missing)
            + ". Re-run without `--no_auto_download_datasets` or place them under the test directory."
        )

    download_root = dataset_root_path / ".downloads" / "swinir_benchmarks"
    _download_swinir_benchmarks(download_root, missing)
    _extract_benchmark_archives(download_root)
    _prepare_from_existing_sources(
        download_root, missing, scale_factor, target_root=dataset_root_path
    )

    missing = _missing_datasets(dataset_root_path, datasets, scale_factor)
    if missing:
        raise RuntimeError(
            "Failed to prepare benchmark datasets: "
            + ", ".join(missing)
            + ". The default source is the public SwinIR benchmark folder: "
            + SWINIR_BENCHMARK_FOLDER_URL
        )


def _missing_datasets(
    dataset_root: Path, datasets: Iterable[str], scale_factor: int
) -> list[str]:
    return [
        name
        for name in datasets
        if not _has_ready_dataset(dataset_root, name, scale_factor)
    ]


def _has_ready_dataset(
    dataset_root: Path, dataset_name: str, scale_factor: int
) -> bool:
    dataset_dir = dataset_root / dataset_name / f"image_SRF_{scale_factor}"
    hr_count = _count_images(dataset_dir / "HR")
    lr_count = _count_images(dataset_dir / "LR")
    return hr_count > 0 and hr_count == lr_count


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(path.suffix.lower() in IMAGE_SUFFIXES for path in directory.iterdir())


def _has_any_image(directory: Path) -> bool:
    return _count_images(directory) > 0


def _prepare_from_existing_sources(
    search_root: Path,
    datasets: Iterable[str],
    scale_factor: int,
    target_root: Path | None = None,
) -> None:
    if target_root is None:
        target_root = search_root

    for dataset_name in datasets:
        if _has_ready_dataset(target_root, dataset_name, scale_factor):
            continue
        source_dir = _find_source_dataset(search_root, dataset_name)
        if source_dir is None:
            continue
        _materialize_dataset(source_dir, target_root, dataset_name, scale_factor)


def _find_source_dataset(search_root: Path, dataset_name: str) -> Path | None:
    aliases = DATASET_ALIASES.get(dataset_name, (dataset_name, dataset_name.lower()))
    if not search_root.exists():
        return None

    for alias in aliases:
        direct = search_root / alias
        if _is_source_dataset_dir(direct):
            return direct

    for candidate in search_root.rglob("*"):
        if not candidate.is_dir() or candidate.name not in aliases:
            continue
        if _is_source_dataset_dir(candidate):
            return candidate
    return None


def _is_source_dataset_dir(path: Path) -> bool:
    try:
        _locate_hr_dir(path)
    except FileNotFoundError:
        return False
    return True


def _locate_hr_dir(path: Path) -> Path:
    candidates = [path / "HR", path / "GTmod12", path / "GT", path]
    for candidate in candidates:
        if _has_any_image(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find HR images under {path}")


def _materialize_dataset(
    source_dir: Path, dataset_root: Path, dataset_name: str, scale_factor: int
) -> None:
    source_hr_dir = _locate_hr_dir(source_dir)
    target_dir = dataset_root / dataset_name / f"image_SRF_{scale_factor}"
    target_hr_dir = target_dir / "HR"
    target_lr_dir = target_dir / "LR"
    target_hr_dir.mkdir(parents=True, exist_ok=True)
    target_lr_dir.mkdir(parents=True, exist_ok=True)

    for source_hr_path in sorted(source_hr_dir.iterdir()):
        if source_hr_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        hr_image = np.array(Image.open(source_hr_path).convert("RGB"))
        hr_image = _modcrop(hr_image, scale_factor)
        lr_image = imresize(hr_image, scale=1.0 / scale_factor)

        Image.fromarray(hr_image).save(target_hr_dir / source_hr_path.name)
        Image.fromarray(lr_image).save(target_lr_dir / source_hr_path.name)


def _modcrop(image: np.ndarray, scale_factor: int) -> np.ndarray:
    height, width = image.shape[:2]
    cropped_height = height - (height % scale_factor)
    cropped_width = width - (width % scale_factor)
    return image[:cropped_height, :cropped_width]


def _extract_benchmark_archives(download_root: Path) -> None:
    for archive_path in download_root.glob("*.zip"):
        extract_target = download_root / archive_path.stem
        if extract_target.exists():
            continue
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(download_root)


def _download_swinir_benchmarks(download_root: Path, datasets: Iterable[str]) -> None:
    download_root.mkdir(parents=True, exist_ok=True)

    import gdown

    for dataset_name in datasets:
        archive_name, extracted_name, file_id = DATASET_ARCHIVES[dataset_name]
        archive_path = download_root / archive_name
        extracted_path = download_root / extracted_name
        if archive_path.exists() or extracted_path.exists():
            continue

        downloaded_path = gdown.download(
            id=file_id, output=str(archive_path), quiet=False
        )
        if downloaded_path is None or not archive_path.exists():
            raise RuntimeError(
                f"Could not download {dataset_name} from the public SwinIR benchmark folder: {SWINIR_BENCHMARK_FOLDER_URL}"
            )
