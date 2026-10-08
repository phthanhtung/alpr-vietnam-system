import React from "react";
// @ts-ignore - CSS imports are provided by the Vite/ViteReact setup.
import "../index.css";
import { useDetection } from "../hooks/useDetection";

export const MinimalLayout: React.FC = () => {
  const { loading, result, error, previewUrl, analyzeImage, reset } =
    useDetection();

  return (
    <div className="minimal-container">
      {/* Header */}
      <header className="minimal-header">
        <h1 className="minimal-title">
          ALPR Vietnam - Hệ Thống Nhận Diện Biển Số
        </h1>
        <span className="status-badge">FastAPI Ready</span>
      </header>

      {/* Toolbar / Actions */}
      <div className="toolbar-actions">
        <input
          type="file"
          accept="image/*"
          onChange={(e) =>
            e.target.files?.[0] && analyzeImage(e.target.files[0])
          }
          className="file-input"
        />
        {previewUrl && (
          <button onClick={reset} className="btn-secondary">
            Làm mới
          </button>
        )}
      </div>

      {/* Trạng thái */}
      {loading && (
        <p className="status-loading">
          Đang thực thi Pipeline YOLOv8 + PaddleOCR...
        </p>
      )}
      {error && <div className="status-error">{error}</div>}

      {/* Kết quả phân tích */}
      {result && (
        <div className="result-grid">
          {/* Cột trái: Ảnh gốc */}
          <div>
            <h3 className="section-label">Ảnh phương tiện</h3>
            <div className="preview-wrapper">
              {previewUrl && (
                <img
                  src={previewUrl}
                  alt="Xe đầu vào"
                  className="image-preview"
                />
              )}
            </div>
          </div>

          {/* Cột phải: Biển số & Benchmark */}
          <div>
            <h3 className="section-label">Kết quả nhận diện</h3>
            {result.plates.map((plate, idx) => (
              <div key={idx} className="plate-card">
                <span className="plate-type-tag">
                  Định dạng:{" "}
                  {plate.plate_type === "single_line"
                    ? "Biển dài 1 dòng"
                    : "Biển vuông 2 dòng"}
                </span>
                <div className="plate-text-display">{plate.plate_text}</div>
                <p className="plate-confidence">
                  Độ tin cậy OCR: {(plate.ocr_confidence * 100).toFixed(1)}%
                </p>
                {plate.crop_base64 && (
                  <div className="crop-preview-wrapper">
                    <span>Ảnh cắt biển số (ROI):</span>
                    <img
                      src={`data:image/jpeg;base64,${plate.crop_base64}`}
                      alt="Crop biển số"
                      className="plate-crop-img"
                    />
                  </div>
                )}
              </div>
            ))}

            {/* Bảng hiệu năng */}
            <div className="benchmark-card">
              <div className="benchmark-title">
                Đo lường thời gian (Benchmark):
              </div>
              <p>
                <span>Tiền xử lý (CLAHE):</span>
                <span>{result.benchmark.preprocess_time_ms.toFixed(1)} ms</span>
              </p>
              <p>
                <span>Suy luận YOLOv8:</span>
                <span>{result.benchmark.inference_time_ms.toFixed(1)} ms</span>
              </p>
              <p>
                <span>Đọc PaddleOCR:</span>
                <span>{result.benchmark.ocr_time_ms.toFixed(1)} ms</span>
              </p>
              <p>
                <span>Tổng độ trễ:</span>
                <span>{result.benchmark.total_time_ms.toFixed(1)} ms</span>
              </p>
              <p className="benchmark-fps">
                <span>Tốc độ khung hình:</span>
                <span>{result.benchmark.fps.toFixed(2)} FPS</span>
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
