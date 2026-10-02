from typing import Tuple, Union
import cv2
import numpy as np


class ImagePreprocessor:
    """Tiền xử lý ảnh: Decode nhị phân và cân bằng sáng thích ứng CLAHE trên kênh LAB."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw bytes sang OpenCV BGR ndarray."""
        if not image_bytes:
            raise ValueError("Dữ liệu ảnh rỗng.")

        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Không thể decode ảnh từ dữ liệu đầu vào.")

        return image

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Áp dụng CLAHE trên kênh L (Luminance) trong không gian màu LAB."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Cân bằng sáng cục bộ trên kênh L (không làm đổi màu sắc)
        enhanced_l = self.clahe.apply(l_channel)

        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def preprocess(self, input_data: Union[bytes, np.ndarray], use_clahe: bool = True) -> np.ndarray:
        """Pipeline tiền xử lý: Chấp nhận bytes hoặc numpy ndarray."""
        if isinstance(input_data, bytes):
            image = self.decode_image(input_data)
        elif isinstance(input_data, np.ndarray):
            image = input_data.copy()
        else:
            raise TypeError("Dữ liệu đầu vào phải là bytes hoặc np.ndarray.")

        if use_clahe:
            image = self.apply_clahe(image)

        return image

    @staticmethod
    def auto_deskew(image: np.ndarray, max_angle: float = 45.0) -> np.ndarray:
        """Tự động ước lượng góc nghiêng và nắn thẳng ảnh biển số (Auto-Deskewing)."""
        if image is None or image.size == 0:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Nhị phân hoá Otsu để tách biệt ký tự và viền biển số
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return image

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Chuẩn hoá góc quay của OpenCV minAreaRect
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # Bỏ qua nếu góc lệch quá nhỏ (< 3 độ) hoặc quá lớn (> max_angle)
        if abs(angle) < 3.0 or abs(angle) > max_angle:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    @classmethod
    def enhance_roi(cls, roi: np.ndarray, target_min_height: int = 100) -> np.ndarray:
        """Tăng cường chất lượng vùng biển số (ROI): Phóng đại + nắn nghiêng."""
        if roi is None or roi.size == 0:
            return roi

        enhanced = roi.copy()
        h, w = enhanced.shape[:2]

        # Nếu ROI quá nhỏ, phóng đại nội suy bậc 3 để làm nét ký tự
        if h < target_min_height:
            scale = target_min_height / float(h)
            enhanced = cv2.resize(enhanced, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Nắn thẳng góc nghiêng
        enhanced = cls.auto_deskew(enhanced)
        return enhanced

