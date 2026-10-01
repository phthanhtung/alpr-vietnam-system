"""Script kiểm thử cục bộ (Test Local CLI) cho ALPR Pipeline

Cho phép chạy thử trực tiếp trên terminal với ảnh mẫu để kiểm tra:
1. Tiền xử lý (CLAHE)
2. Suy luận YOLOv8 (best.pt)
3. Cắt vùng biển số (ROI Crop)
4. Nhận diện ký tự PaddleOCR & ghép dòng biển số Việt Nam
5. Báo cáo đo lường độ trễ (ms) và FPS
"""

import argparse
from pathlib import Path
import sys

# Đảm bảo import được module trong backend/src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))

from src.services.pipeline import ALPRPipeline


def main():
    parser = argparse.ArgumentParser(description="Chạy thử nghiệm ALPR Vietnam Pipeline trên ảnh cục bộ")
    parser.add_argument(
        "--image",
        type=str,
        default=str(project_root / "data" / "sample" / "images" / "BienSoXe-20-_jpg.rf.5ea0fd8c424168c8364ffa85f4c525b2.jpg"),
        help="Đường dẫn tới file ảnh cần kiểm tra"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(current_dir / "models" / "best.pt"),
        help="Đường dẫn file trọng số YOLOv8 (.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Ngưỡng tin cậy (Confidence threshold, mặc định 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Ngưỡng NMS IoU (mặc định 0.45)"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Tắt bước nhận diện OCR (chỉ chạy YOLO Object Detection)"
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Tắt bước tiền xử lý cân bằng sáng CLAHE"
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"\n❌ LỖI: Không tìm thấy file ảnh tại: {image_path}")
        print(f"👉 Gợi ý: Hãy kiểm tra các file ảnh trong: {project_root / 'data' / 'sample' / 'images'}")
        return

    print("=" * 65)
    print("🚀 BẮT ĐẦU KIỂM THỬ ALPR VIETNAM PIPELINE (YOLOv8 + PaddleOCR)")
    print("=" * 65)
    print(f"📷 File ảnh:    {image_path.name}")
    print(f"🎯 Trọng số:    {Path(args.model).name}")
    print(f"⚙️  Ngưỡng:      conf={args.conf} | iou={args.iou}")
    print(f"🔧 Cấu hình:    CLAHE={'Tắt' if args.no_clahe else 'Bật'} | OCR={'Tắt' if args.no_ocr else 'Bật'}")
    print("-" * 65)

    # 1. Đọc dữ liệu ảnh nhị phân
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # 2. Khởi tạo Pipeline
    pipeline = ALPRPipeline(
        model_path=str(args.model),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device="cpu"
    )

    # 3. Thực thi Pipeline toàn trình
    output = pipeline.process(
        input_data=img_bytes,
        conf=args.conf,
        iou=args.iou,
        use_clahe=not args.no_clahe,
        enable_ocr=not args.no_ocr,
        encode_crop=True
    )

    # 4. In kết quả nhận diện
    print("\n📋 KẾT QUẢ NHẬN DIỆN BIỂN SỐ XE:")
    print(f"🔹 Tổng số biển số phát hiện được: {output.total_plates_found}")

    if output.total_plates_found == 0:
        print("⚠️  Không phát hiện thấy biển số nào trong ảnh này. (Thử giảm --conf xuống 0.15 xem sao).")
    else:
        for idx, plate in enumerate(output.plates, 1):
            print(f"\n   🚗 [Biển số #{idx}]")
            print(f"   ├─ Tọa độ Box (x1,y1,x2,y2): [{plate.box.x1}, {plate.box.y1}, {plate.box.x2}, {plate.box.y2}]")
            print(f"   ├─ Độ tin cậy phát hiện (YOLO): {plate.detection_confidence * 100:.2f}%")
            if plate.plate_text:
                print(f"   ├─ 🏷️  KÝ TỰ BIỂN SỐ ĐỌC ĐƯỢC: {plate.plate_text}")
                print(f"   ├─ Độ tin cậy ký tự (OCR):    {plate.ocr_confidence * 100:.2f}%" if plate.ocr_confidence else "   ├─ Độ tin cậy OCR: N/A")
                print(f"   ├─ Phân loại dạng biển:       {plate.plate_type} (Chi tiết: {plate.line_details})")
                print(f"   └─ Chuỗi Base64 ảnh crop:     {plate.crop_base64[:30]}... (Độ dài: {len(plate.crop_base64)} ký tự)")
            else:
                print("   └─ Ký tự biển số: (Chưa bật OCR)")

    # 5. In bảng số liệu hiệu năng (Benchmark)
    bm = output.benchmark
    print("\n" + "=" * 65)
    print("⏱️  BÁO CÁO HIỆU NĂNG THỰC NGHIỆM (LATENCY BENCHMARK)")
    print("=" * 65)
    print(f" 1. Tiền xử lý (CLAHE):          {bm.preprocess_time_ms:>8.2f} ms")
    print(f" 2. Suy luận YOLOv8 (Inference): {bm.inference_time_ms:>8.2f} ms")
    print(f" 3. Hậu xử lý & Cắt ROI (Crop):  {bm.postprocess_time_ms:>8.2f} ms")
    print(f" 4. Nhận diện PaddleOCR:         {bm.ocr_time_ms:>8.2f} ms")
    print(f" ----------------------------------------------")
    print(f" 🌟 Tổng thời gian (End-to-End): {bm.total_time_ms:>8.2f} ms")
    print(f" ⚡ Tốc độ khung hình (FPS):     {bm.fps:>8.2f} FPS")
    print("=" * 65)


if __name__ == "__main__":
    main()
