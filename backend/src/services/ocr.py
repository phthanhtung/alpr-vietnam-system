from dataclasses import dataclass
import re
from typing import Any, Dict, List, Tuple
import numpy as np
from paddleocr import PaddleOCR

from .preprocessing import ImagePreprocessor


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
    """Nhận diện ký tự biển số bằng PaddleOCR và phân loại / định dạng theo chuẩn Biển số Việt Nam."""

    # Bảng ánh xạ khử nhầm lẫn quang học (Disambiguation Mappings)
    CHAR_TO_NUM = {
        'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1',
        'Z': '2', 'A': '4', 'S': '5', 'B': '8', 'G': '6'
    }
    NUM_TO_CHAR = {
        '0': 'D', '8': 'B', '1': 'T', '5': 'S',
        '2': 'Z', '4': 'A', '6': 'G'
    }

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

    @classmethod
    def format_vietnam_plate(cls, line_texts: List[str], plate_type: str) -> Tuple[str, str, List[str]]:
        """Định dạng và sửa lỗi ký tự theo quy chuẩn Biển số xe Việt Nam (Thông tư 24/2023/TT-BCA)."""
        if not line_texts:
            return "", plate_type, []

        if plate_type == "multi_line" or len(line_texts) >= 2:
            l1 = re.sub(r"[^A-Z0-9]", "", line_texts[0].upper())
            l2 = re.sub(r"[^A-Z0-9]", "", line_texts[1].upper()) if len(line_texts) > 1 else ""

            # Dòng 1: [2 số tỉnh] + [1 chữ sê-ri] + [1 số/chữ tuỳ chọn] (Ví dụ: 47-D1, 59-K2, 29-AA)
            if len(l1) >= 3:
                prov = "".join(cls.CHAR_TO_NUM.get(c, c) for c in l1[:2])
                series = cls.NUM_TO_CHAR.get(l1[2], l1[2]) if l1[2].isdigit() else l1[2]
                tail_l1 = l1[3:]
                l1_fixed = f"{prov}-{series}{tail_l1}"
            else:
                l1_fixed = l1

            # Dòng 2: Toàn bộ là số (Ví dụ: 221.56 hoặc 9672)
            l2_digits = "".join(cls.CHAR_TO_NUM.get(c, c) for c in l2)
            if len(l2_digits) == 5:
                l2_fixed = f"{l2_digits[:3]}.{l2_digits[3:]}"
            elif len(l2_digits) == 4:
                l2_fixed = l2_digits
            else:
                l2_fixed = l2_digits

            full_plate = f"{l1_fixed}-{l2_fixed}" if (l1_fixed and l2_fixed) else f"{l1_fixed}{l2_fixed}"
            return full_plate, "multi_line", [l1_fixed, l2_fixed]

        else:
            # Biển dài 1 dòng: [2 số tỉnh][1-2 chữ sê-ri]-[4-5 số xe] (Ví dụ: 30F-557.75)
            clean = re.sub(r"[^A-Z0-9]", "", line_texts[0].upper())
            if len(clean) >= 7:
                prov = "".join(cls.CHAR_TO_NUM.get(c, c) for c in clean[:2])
                series = cls.NUM_TO_CHAR.get(clean[2], clean[2]) if clean[2].isdigit() else clean[2]
                tail = "".join(cls.CHAR_TO_NUM.get(c, c) for c in clean[3:])

                if len(tail) == 5:
                    full_plate = f"{prov}{series}-{tail[:3]}.{tail[3:]}"
                elif len(tail) == 4:
                    full_plate = f"{prov}{series}-{tail}"
                else:
                    full_plate = f"{prov}{series}-{tail}"
                return full_plate, "single_line", [full_plate]

            return clean, "single_line", [clean]

    def _sort_and_merge_lines(
        self,
        ocr_data: List[Tuple[List[List[float]], Tuple[str, float]]]
    ) -> Tuple[str, float, str, List[str]]:
        """Thuật toán sắp xếp không gian (Spatial Sorting) và áp dụng quy chuẩn biển số Việt Nam."""
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

        # 1 cụm chữ duy nhất -> Kiểm tra biển 1 dòng
        if len(boxes_with_info) == 1:
            raw_text = boxes_with_info[0]["text"]
            formatted_text, p_type, lines = self.format_vietnam_plate([raw_text], "single_line")
            return formatted_text, round(avg_conf, 4), p_type, lines

        # Phân tích độ lệch trục Y để phân loại biển 1 dòng vs 2 dòng
        y_centers = [item["cy"] for item in boxes_with_info]
        y_range = max(y_centers) - min(y_centers)
        avg_h = np.mean([item["h"] for item in boxes_with_info])

        if y_range > (avg_h * 0.4):
            # Biển vuông 2 dòng
            median_y = (max(y_centers) + min(y_centers)) / 2.0

            line1 = [item for item in boxes_with_info if item["cy"] <= median_y]
            line2 = [item for item in boxes_with_info if item["cy"] > median_y]

            line1_sorted = sorted(line1, key=lambda x: x["cx"])
            line2_sorted = sorted(line2, key=lambda x: x["cx"])

            line1_text = "".join(item["text"] for item in line1_sorted)
            line2_text = "".join(item["text"] for item in line2_sorted)

            formatted_text, p_type, lines = self.format_vietnam_plate(
                [line1_text, line2_text],
                plate_type="multi_line"
            )
            return formatted_text, round(avg_conf, 4), p_type, lines

        else:
            # Biển dài 1 dòng
            sorted_boxes = sorted(boxes_with_info, key=lambda x: x["cx"])
            full_text = "".join(item["text"] for item in sorted_boxes)
            formatted_text, p_type, lines = self.format_vietnam_plate([full_text], plate_type="single_line")
            return formatted_text, round(avg_conf, 4), p_type, lines

    @staticmethod
    def _calculate_box_angle(box: List[List[float]]) -> float:
        """Tính góc nghiêng của polygon từ điểm p0 sang p1."""
        if len(box) < 2:
            return 0.0
        dx = box[1][0] - box[0][0]
        dy = box[1][1] - box[0][1]
        import math
        return float(math.degrees(math.atan2(dy, dx)))

    def recognize(self, cropped_plate_img: np.ndarray, auto_deskew: bool = True) -> OCRResult:
        """Nhận diện chữ số từ ảnh crop biển số kèm tự động nắn nghiêng 2-Pass và hậu xử lý."""
        if cropped_plate_img is None or cropped_plate_img.size == 0:
            return OCRResult(plate_text="", confidence=0.0, plate_type="unknown", line_details=[])

        try:
            # Pass 1: Nhận diện trực tiếp trên ROI
            ocr_results = self.ocr.ocr(cropped_plate_img, cls=True)
            chosen_data = ocr_results[0] if (ocr_results and ocr_results[0]) else []

            # Pass 2 (Adaptive Deskewing): Nếu phát hiện chữ bị nghiêng, xoay phẳng và quét lại
            if auto_deskew and chosen_data:
                angles = [self._calculate_box_angle(item[0]) for item in chosen_data]
                if angles:
                    median_angle = float(np.median(angles))
                    if 3.0 < abs(median_angle) < 60.0:
                        (h, w) = cropped_plate_img.shape[:2]
                        center = (w // 2, h // 2)
                        import cv2
                        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        rotated = cv2.warpAffine(
                            cropped_plate_img,
                            M,
                            (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE
                        )
                        pass2_res = self.ocr.ocr(rotated, cls=True)
                        if pass2_res and pass2_res[0]:
                            # Ưu tiên Pass 2 nếu đọc được nhiều thông tin hơn hoặc độ tin cậy cao hơn
                            chosen_data = pass2_res[0]

            if not chosen_data:
                return OCRResult(plate_text="", confidence=0.0, plate_type="unknown", line_details=[])

            plate_text, conf, p_type, lines = self._sort_and_merge_lines(chosen_data)
            return OCRResult(plate_text=plate_text, confidence=conf, plate_type=p_type, line_details=lines)
        except Exception:
            return OCRResult(plate_text="", confidence=0.0, plate_type="error", line_details=[])

