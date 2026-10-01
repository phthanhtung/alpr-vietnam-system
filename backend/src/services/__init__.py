"""Package Services: Chứa toàn bộ các module xử lý Computer Vision & OCR cốt lõi của Tầng 1."""

from .preprocessing import ImagePreprocessor
from .detector import YOLOPlateDetector
from .postprocessing import PostProcessor, BoundingBox, RawDetection
from .ocr import PaddleOCREngine, OCRResult
from .benchmark import LatencyBenchmark, BenchmarkReport
from .pipeline import ALPRPipeline, CompletePlateResult, PipelineOutput

__all__ = [
    "ImagePreprocessor",
    "YOLOPlateDetector",
    "PostProcessor",
    "BoundingBox",
    "RawDetection",
    "PaddleOCREngine",
    "OCRResult",
    "LatencyBenchmark",
    "BenchmarkReport",
    "ALPRPipeline",
    "CompletePlateResult",
    "PipelineOutput",
]
