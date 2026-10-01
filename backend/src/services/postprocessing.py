import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2)
        }


@dataclass
class RawDetection:
    box: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    cropped_image: Optional[np.ndarray] = field(default=None, repr=False)
    crop_base64: Optional[str] = None


class PostProcessor:
    """Hậu xử lý: Bóc tách Bounding Box, cắt ROI an toàn và encode Base64."""

    def __init__(self, crop_padding: int = 4):
        self.crop_padding = crop_padding

    def crop_roi(self, image: np.ndarray, box: BoundingBox, padding: Optional[int] = None) -> np.ndarray:
        """Cắt ảnh biển số (ROI) kèm kiểm tra biên (Boundary Check) và Padding."""
        pad = padding if padding is not None else self.crop_padding
        height, width = image.shape[:2]

        x1 = max(0, int(box.x1) - pad)
        y1 = max(0, int(box.y1) - pad)
        x2 = min(width, int(box.x2) + pad)
        y2 = min(height, int(box.y2) + pad)

        return image[y1:y2, x1:x2]

    @staticmethod
    def encode_image_to_base64(image: np.ndarray, image_format: str = ".jpg") -> str:
        """Chuyển ma trận ảnh NumPy sang chuỗi Base64."""
        if image is None or image.size == 0:
            return ""
        success, buffer = cv2.imencode(image_format, image)
        return base64.b64encode(buffer).decode("utf-8") if success else ""

    def extract_detections(
        self,
        yolo_results: List[Any],
        original_image: np.ndarray,
        encode_crop: bool = True
    ) -> List[RawDetection]:
        """Trích xuất danh sách biển số và ảnh crop từ kết quả YOLOv8."""
        detections: List[RawDetection] = []
        if not yolo_results or not yolo_results[0].boxes:
            return detections

        result = yolo_results[0]
        names_dict = getattr(result, "names", {})

        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            cls_name = names_dict.get(cls_id, "license_plate")

            bbox = BoundingBox(
                x1=round(xyxy[0], 2),
                y1=round(xyxy[1], 2),
                x2=round(xyxy[2], 2),
                y2=round(xyxy[3], 2)
            )

            cropped = self.crop_roi(original_image, bbox)
            crop_b64 = self.encode_image_to_base64(cropped) if (encode_crop and cropped.size > 0) else None

            detections.append(
                RawDetection(
                    box=bbox,
                    confidence=round(conf, 4),
                    class_id=cls_id,
                    class_name=cls_name,
                    cropped_image=cropped,
                    crop_base64=crop_b64
                )
            )

        return detections
