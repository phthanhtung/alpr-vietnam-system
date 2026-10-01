from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .benchmark import BenchmarkReport, LatencyBenchmark
from .detector import YOLOPlateDetector
from .ocr import OCRResult, PaddleOCREngine
from .postprocessing import BoundingBox, PostProcessor, RawDetection
from .preprocessing import ImagePreprocessor


@dataclass
class CompletePlateResult:
    """Kết quả hoàn chỉnh của 1 biển số xe sau Detector và OCR."""
    box: BoundingBox
    detection_confidence: float
    class_id: int
    class_name: str
    plate_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    plate_type: Optional[str] = None
    line_details: List[str] = field(default_factory=list)
    crop_base64: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "detection_confidence": self.detection_confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "plate_text": self.plate_text,
            "ocr_confidence": self.ocr_confidence,
            "plate_type": self.plate_type,
            "line_details": self.line_details,
            "crop_base64": self.crop_base64
        }


@dataclass
class PipelineOutput:
    """Đầu ra tổng thể của ALPR Pipeline."""
    success: bool
    total_plates_found: int
    plates: List[CompletePlateResult]
    benchmark: BenchmarkReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_plates_found": self.total_plates_found,
            "plates": [p.to_dict() for p in self.plates],
            "benchmark": self.benchmark.to_dict()
        }


class ALPRPipeline:
    """Điều phối chuỗi xử lý End-to-End: Preprocess -> Detect -> Crop -> OCR -> Benchmark."""

    def __init__(
        self,
        model_path: str = "models/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        lazy_ocr: bool = False
    ):
        self.preprocessor = ImagePreprocessor()
        self.detector = YOLOPlateDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        self.postprocessor = PostProcessor()
        self.ocr_engine: Optional[PaddleOCREngine] = None

        if not lazy_ocr:
            self._init_ocr()

    def _init_ocr(self) -> None:
        """Khởi tạo PaddleOCR Engine."""
        if self.ocr_engine is None:
            try:
                self.ocr_engine = PaddleOCREngine(use_angle_cls=True, lang="en")
            except Exception as e:
                print(f"[ALPRPipeline] Cảnh báo không thể nạp PaddleOCR: {e}")

    def process(
        self,
        input_data: Union[bytes, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        use_clahe: bool = True,
        enable_ocr: bool = True,
        encode_crop: bool = True
    ) -> PipelineOutput:
        """Thực thi chuỗi xử lý toàn trình từ ảnh đầu vào."""
        benchmark = LatencyBenchmark()
        benchmark.start()

        # 1. Tiền xử lý (CLAHE)
        processed_image = self.preprocessor.preprocess(input_data, use_clahe=use_clahe)
        benchmark.record_preprocess()

        # 2. Suy luận YOLOv8
        raw_results = self.detector.detect(processed_image, conf=conf, iou=iou)
        benchmark.record_inference()

        # 3. Hậu xử lý & Cắt ROI
        detections: List[RawDetection] = self.postprocessor.extract_detections(
            raw_results,
            processed_image,
            encode_crop=encode_crop
        )
        benchmark.record_postprocess()

        # 4. Nhận diện ký tự OCR
        complete_results: List[CompletePlateResult] = []

        if enable_ocr and detections:
            if self.ocr_engine is None:
                self._init_ocr()

            for det in detections:
                ocr_res = self.ocr_engine.recognize(det.cropped_image) if self.ocr_engine else None

                complete_results.append(
                    CompletePlateResult(
                        box=det.box,
                        detection_confidence=det.confidence,
                        class_id=det.class_id,
                        class_name=det.class_name,
                        plate_text=ocr_res.plate_text if ocr_res else "",
                        ocr_confidence=ocr_res.confidence if ocr_res else 0.0,
                        plate_type=ocr_res.plate_type if ocr_res else "unknown",
                        line_details=ocr_res.line_details if ocr_res else [],
                        crop_base64=det.crop_base64
                    )
                )
        else:
            for det in detections:
                complete_results.append(
                    CompletePlateResult(
                        box=det.box,
                        detection_confidence=det.confidence,
                        class_id=det.class_id,
                        class_name=det.class_name,
                        plate_text=None,
                        ocr_confidence=None,
                        plate_type=None,
                        line_details=[],
                        crop_base64=det.crop_base64
                    )
                )

        benchmark.record_ocr()

        return PipelineOutput(
            success=True,
            total_plates_found=len(complete_results),
            plates=complete_results,
            benchmark=benchmark.get_report()
        )
