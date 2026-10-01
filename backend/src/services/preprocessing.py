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
