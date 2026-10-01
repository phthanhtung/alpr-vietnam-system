from pathlib import Path
from typing import Any, List, Optional
import numpy as np
from ultralytics import YOLO


class YOLOPlateDetector:
    """Suy luận YOLOv8 phát hiện vị trí biển số xe."""

    def __init__(
        self,
        model_path: str = "models/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model: Optional[YOLO] = None

        self._load_model()

    def _load_model(self) -> None:
        """Tải mô hình vào bộ nhớ RAM/VRAM."""
        if self.model_path.exists():
            self.model = YOLO(str(self.model_path))
        else:
            self.model = YOLO("yolov8n.pt")

    def detect(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[Any]:
        """Dự đoán Bounding Box trên ảnh với ngưỡng conf và NMS iou tùy chỉnh."""
        if self.model is None:
            raise RuntimeError("Mô hình YOLOv8 chưa được khởi tạo.")

        active_conf = conf if conf is not None else self.conf_threshold
        active_iou = iou if iou is not None else self.iou_threshold

        return self.model.predict(
            source=image,
            conf=active_conf,
            iou=active_iou,
            device=self.device,
            verbose=False
        )
