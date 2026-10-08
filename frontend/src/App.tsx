import React from "react";
import { useDetection } from "./hooks/useDetection";

export const MinimalLayout: React.FC = () => {
  const { loading, result, error, previewUrl, analyzeImage, reset } =
    useDetection();

  return (
    <main className="minimal-container">
      <header className="minimal-header">
        <h1 className="minimal-title">
          ALPR Vietnam - Hệ Thống Nhận Diện Biển Số
        </h1>
        <span className="status-badge">
          v1.0 Ready
        </span>
      </header>

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
          <button
            onClick={reset}
            className="btn-secondary"
          >
            Làm mới
          </button>
        )}
      </div>

      {loading && (
        <p className="status-loading">
          Đang chạy Pipeline YOLOv8 + PaddleOCR...
        </p>
      )}
      {error && <p className="status-error">{error}</p>}

      {result && (
        <div className="result-grid">
          <div>
            <h2 className="section-label">Ảnh gốc</h2>
            {previewUrl && (
              <div className="preview-wrapper">
                <img src={previewUrl} alt="Ảnh đã chọn" className="image-preview" />
              </div>
            )}
          </div>
          <div>
            <h2 className="section-label">Kết quả phân tích</h2>
            {result.plates.map((plate, idx) => (
              <article key={idx} className="plate-card">
                <span className="plate-type-tag">
                  Loại: {plate.plate_type}
                </span>
                <div className="plate-text-display">
                  {plate.plate_text}
                </div>
                <p className="plate-confidence">
                  Độ tin cậy OCR: {(plate.ocr_confidence * 100).toFixed(1)}%
                </p>
                {plate.crop_base64 && (
                  <div className="crop-preview-wrapper">
                    <span>Ảnh biển số</span>
                    <img
                      src={`data:image/jpeg;base64,${plate.crop_base64}`}
                      alt={`Biển số ${plate.plate_text}`}
                      className="plate-crop-img"
                    />
                  </div>
                )}
              </article>
            ))}

            <div className="benchmark-card">
              <p className="benchmark-title">Hiệu năng</p>
              <p>
                <span>Suy luận YOLO</span>
                <span>{result.benchmark.inference_time_ms.toFixed(1)} ms</span>
              </p>
              <p>
                <span>Đọc PaddleOCR</span>
                <span>{result.benchmark.ocr_time_ms.toFixed(1)} ms</span>
              </p>
              <p className="benchmark-fps">
                <span>Tốc độ</span>
                <span>{result.benchmark.fps.toFixed(2)} FPS</span>
              </p>
            </div>
          </div>
        </div>
      )}
    </main>
  );
};
