"""Pass/fail inspection logic dựa trên OpenCV.

Hiện hỗ trợ:
- TemplateMatchInspector: so sánh ảnh với golden sample (whole-image).
- BrightnessInspector: kiểm tra độ sáng trung bình trong ROI có trong khoảng [min,max].
- MultiTemplateInspector: tìm vật thể cụ thể (đếm / phân loại / vị trí) bằng
  template matching + Non-Max Suppression.

CompositeInspector gộp nhiều inspector, kết quả PASS chỉ khi TẤT CẢ đều PASS.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# NOTE: KHÔNG import MultiTemplateInspector ở top — sẽ gây circular import
# (multi_template.py cũng import BaseInspector từ inspector.py). Import lazy
# trong build_inspector() bên dưới, lúc đó inspector.py đã load xong.


@dataclass(frozen=True, slots=True)
class InspectionResult:
    name: str
    passed: bool
    score: float
    detail: str = ""


@dataclass(frozen=True, slots=True)
class InspectionReport:
    results: tuple[InspectionResult, ...]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)


class BaseInspector(ABC):
    name: str = "Base"

    @abstractmethod
    def inspect(self, image: np.ndarray) -> InspectionReport: ...


class TemplateMatchInspector(BaseInspector):
    """So khớp ảnh chụp với reference bằng cv2.matchTemplate (TM_CCOEFF_NORMED).

    Score ∈ [-1, 1]; score ≥ threshold → PASS.
    """

    name = "TemplateMatch"

    def __init__(self, reference_path: str | Path, threshold: float = 0.9) -> None:
        ref = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise FileNotFoundError(f"Không đọc được reference: {reference_path}")
        self._reference = ref
        self._threshold = float(threshold)

    def inspect(self, image: np.ndarray) -> InspectionReport:
        gray = _to_gray(image)
        rh, rw = self._reference.shape[:2]
        ih, iw = gray.shape[:2]
        if ih < rh or iw < rw:
            return InspectionReport(
                results=(
                    InspectionResult(
                        name=self.name,
                        passed=False,
                        score=0.0,
                        detail=f"Reference {rw}x{rh} lớn hơn ảnh {iw}x{ih}",
                    ),
                )
            )
        result = cv2.matchTemplate(gray, self._reference, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        passed = score >= self._threshold
        return InspectionReport(
            results=(
                InspectionResult(
                    name=self.name,
                    passed=passed,
                    score=float(score),
                    detail=f"score={score:.4f} threshold={self._threshold}",
                ),
            )
        )


class BrightnessInspector(BaseInspector):
    """Kiểm tra mean brightness trong ROI nằm trong khoảng [min_val, max_val]."""

    name = "Brightness"

    def __init__(
        self,
        roi: tuple[int, int, int, int],
        min_val: float,
        max_val: float = 255.0,
    ) -> None:
        self._roi = roi
        self._min = float(min_val)
        self._max = float(max_val)

    def inspect(self, image: np.ndarray) -> InspectionReport:
        x, y, w, h = self._roi
        crop = image[y : y + h, x : x + w]
        gray = _to_gray(crop)
        mean = float(np.mean(gray))
        passed = self._min <= mean <= self._max
        return InspectionReport(
            results=(
                InspectionResult(
                    name=self.name,
                    passed=passed,
                    score=mean,
                    detail=f"mean={mean:.2f} range=[{self._min:.0f},{self._max:.0f}]",
                ),
            )
        )


class CompositeInspector(BaseInspector):
    """Chạy nhiều inspector, AND kết quả. PASS khi tất cả PASS."""

    name = "Composite"

    def __init__(self, inspectors: list[BaseInspector]) -> None:
        self._inspectors = inspectors

    def inspect(self, image: np.ndarray) -> InspectionReport:
        results: list[InspectionResult] = []
        for insp in self._inspectors:
            results.extend(insp.inspect(image).results)
        return InspectionReport(results=tuple(results))


def _as_roi(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return tuple(int(v) for v in value)  # type: ignore[return-value]


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert ảnh sang gray, xử lý được cả 1-channel (mono) và 3-channel (BGR)."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def build_inspector(cfg: dict) -> CompositeInspector:
    """Build CompositeInspector từ config dict.

    Hỗ trợ 2 mode:
      A) Multi-object (mới): config có "templates" → MultiTemplateInspector.
         Thêm các key tuỳ chọn: expected_count, expected_template, expected_position.
      B) Legacy (cũ): config có "reference_image" → TemplateMatchInspector.
    """
    # Lazy import để tránh circular: multi_template.py import từ inspector.py,
    # nên inspector.py không được import multi_template ở top-level.
    from multi_template import MultiTemplateInspector

    inspectors: list[BaseInspector] = []
    if cfg.get("templates"):
        # Mode A: tìm vật thể cụ thể.
        expected_position = cfg.get("expected_position")
        if expected_position is not None:
            expected_position = (int(expected_position[0]), int(expected_position[1]))
        inspectors.append(
            MultiTemplateInspector(
                templates=cfg["templates"],
                threshold=float(cfg.get("object_score_threshold", 0.85)),
                nms_radius=int(cfg.get("nms_radius", 20)),
                min_count=cfg.get("expected_count"),
                expected_template=cfg.get("expected_template"),
                expected_position=expected_position,
                position_tolerance=int(cfg.get("position_tolerance", 30)),
            )
        )
    elif cfg.get("reference_image"):
        # Mode B: backward compat.
        inspectors.append(
            TemplateMatchInspector(
                cfg["reference_image"],
                threshold=float(cfg.get("template_match_threshold", 0.9)),
            )
        )
    roi = _as_roi(cfg.get("roi"))
    if roi is not None and "min_brightness" in cfg:
        inspectors.append(
            BrightnessInspector(
                roi,
                min_val=float(cfg["min_brightness"]),
                max_val=float(cfg.get("max_brightness", 255.0)),
            )
        )
    if not inspectors:
        raise ValueError(
            "inspection.* trong config không có check nào. "
            "Cần 'templates' (multi-object) HOẶC 'reference_image' (legacy) "
            "HOẶC ('roi' + 'min_brightness')."
        )
    return CompositeInspector(inspectors)
