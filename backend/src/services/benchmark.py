from dataclasses import dataclass
import time
from typing import Dict


@dataclass
class BenchmarkReport:
    """Báo cáo chi tiết độ trễ từng chặng (ms) và FPS."""
    preprocess_time_ms: float
    inference_time_ms: float
    postprocess_time_ms: float
    ocr_time_ms: float
    total_time_ms: float
    fps: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "preprocess_time_ms": self.preprocess_time_ms,
            "inference_time_ms": self.inference_time_ms,
            "postprocess_time_ms": self.postprocess_time_ms,
            "ocr_time_ms": self.ocr_time_ms,
            "total_time_ms": self.total_time_ms,
            "fps": self.fps
        }


class LatencyBenchmark:
    """Đo lường thời gian thực thi (ms) từng chặng và tính FPS."""

    def __init__(self):
        self._t_start: float = 0.0
        self._t_mark: float = 0.0
        self.t_pre: float = 0.0
        self.t_infer: float = 0.0
        self.t_post: float = 0.0
        self.t_ocr: float = 0.0

    def start(self) -> None:
        self._t_start = time.perf_counter()
        self._t_mark = self._t_start

    def record_preprocess(self) -> None:
        now = time.perf_counter()
        self.t_pre = (now - self._mark_check()) * 1000.0
        self._t_mark = now

    def record_inference(self) -> None:
        now = time.perf_counter()
        self.t_infer = (now - self._mark_check()) * 1000.0
        self._t_mark = now

    def record_postprocess(self) -> None:
        now = time.perf_counter()
        self.t_post = (now - self._mark_check()) * 1000.0
        self._t_mark = now

    def record_ocr(self) -> None:
        now = time.perf_counter()
        self.t_ocr = (now - self._mark_check()) * 1000.0
        self._t_mark = now

    def _mark_check(self) -> float:
        return self._t_mark if self._t_mark > 0 else time.perf_counter()

    def get_report(self) -> BenchmarkReport:
        total_time_ms = self.t_pre + self.t_infer + self.t_post + self.t_ocr
        fps = (1000.0 / total_time_ms) if total_time_ms > 0 else 0.0

        return BenchmarkReport(
            preprocess_time_ms=round(self.t_pre, 2),
            inference_time_ms=round(self.t_infer, 2),
            postprocess_time_ms=round(self.t_post, 2),
            ocr_time_ms=round(self.t_ocr, 2),
            total_time_ms=round(total_time_ms, 2),
            fps=round(fps, 2)
        )
