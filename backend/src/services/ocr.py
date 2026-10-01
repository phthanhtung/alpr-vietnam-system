from dataclasses import dataclass
import re
from typing import Any, Dict, List, Tuple
import numpy as np
from paddleocr import PaddleOCR


@dataclass
class OCRResult:
    """Kết quả nhận diện ký tự biển số xe."""
    plate_text: str
    confidence: float
    plate_type: str  # "single_line" | "multi_line" | "unknown"
    line_details: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plate_text": self.plate_text,
            "confidence": self.confidence,
            "plate_type": self.plate_type,
            "line_details": self.line_details
        }


class PaddleOCREngine:
    """Nhận diện ký tự biển số bằng PaddleOCR và phân loại biển 1 dòng / 2 dòng."""

    def __init__(self, use_angle_cls: bool = True, lang: str = "en"):
        try:
            self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)
        except Exception:
            self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    @staticmethod
    def clean_text(text: str) -> str:
        """Lọc bỏ ký tự rác bằng regex, chỉ giữ chữ in hoa, số, chấm và gạch ngang."""
        text = text.upper().strip()
        return re.sub(r"[^A-Z0-9\.\-]", "", text)

    def _sort_and_merge_lines(
        self,
        ocr_data: List[Tuple[List[List[float]], Tuple[str, float]]]
    ) -> Tuple[str, float, str, List[str]]:
        """Thuật toán sắp xếp không gian (Spatial Sorting) cho biển số Việt Nam."""
        if not ocr_data:
            return "", 0.0, "unknown", []

        boxes_with_info = []
        total_conf = 0.0

        for item in ocr_data:
            box = item[0]
            text, conf = item[1]

            cleaned = self.clean_text(text)
            if not cleaned:
                continue

            pts = np.array(box, dtype=np.float32)
            center_y = float(np.mean(pts[:, 1]))
            center_x = float(np.mean(pts[:, 0]))
            box_h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))

            boxes_with_info.append({
                "text": cleaned,
                "conf": float(conf),
                "cy": center_y,
                "cx": center_x,
                "h": box_h
            })
            total_conf += float(conf)

        if not boxes_with_info:
            return "", 0.0, "unknown", []

        avg_conf = total_conf / len(boxes_with_info)

        # 1 cụm chữ duy nhất -> Biển 1 dòng
        if len(boxes_with_info) == 1:
            plate_text = boxes_with_info[0]["text"]
            return plate_text, round(avg_conf, 4), "single_line", [plate_text]

        # Phân tích độ lệch trục Y để phân loại biển 1 dòng vs 2 dòng
        y_centers = [item["cy"] for item in boxes_with_info]
        y_range = max(y_centers) - min(y_centers)
        avg_h = np.mean([item["h"] for item in boxes_with_info])

        if y_range > (avg_h * 0.5):
            # Biển vuông 2 dòng
            plate_type = "multi_line"
            median_y = (max(y_centers) + min(y_centers)) / 2.0

            line1 = [item for item in boxes_with_info if item["cy"] <= median_y]
            line2 = [item for item in boxes_with_info if item["cy"] > median_y]

            line1_sorted = sorted(line1, key=lambda x: x["cx"])
            line2_sorted = sorted(line2, key=lambda x: x["cx"])

            line1_text = "".join(item["text"] for item in line1_sorted)
            line2_text = "".join(item["text"] for item in line2_sorted)

            plate_text = f"{line1_text}-{line2_text}" if line1_text and line2_text else f"{line1_text}{line2_text}"
            return plate_text, round(avg_conf, 4), plate_type, [line1_text, line2_text]

        else:
            # Biển dài 1 dòng
            plate_type = "single_line"
            sorted_boxes = sorted(boxes_with_info, key=lambda x: x["cx"])
            full_text = "".join(item["text"] for item in sorted_boxes)
            return full_text, round(avg_conf, 4), plate_type, [full_text]

    def recognize(self, cropped_plate_img: np.ndarray) -> OCRResult:
        """Nhận diện chữ số từ ảnh crop biển số."""
        if cropped_plate_img is None or cropped_plate_img.size == 0:
            return OCRResult(plate_text="", confidence=0.0, plate_type="unknown", line_details=[])

        try:
            ocr_results = self.ocr.ocr(cropped_plate_img, cls=True)
            if not ocr_results or ocr_results[0] is None:
                return OCRResult(plate_text="", confidence=0.0, plate_type="unknown", line_details=[])

            plate_text, conf, p_type, lines = self._sort_and_merge_lines(ocr_results[0])
            return OCRResult(plate_text=plate_text, confidence=conf, plate_type=p_type, line_details=lines)
        except Exception:
            return OCRResult(plate_text="", confidence=0.0, plate_type="error", line_details=[])
