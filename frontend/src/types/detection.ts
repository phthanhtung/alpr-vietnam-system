export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface PlateResult {
  box: BoundingBox;
  detection_confidence: number;
  class_id: number;
  class_name: string;
  plate_text: string;
  ocr_confidence: number;
  plate_type: 'single_line' | 'multi_line';
  line_details: string[];
  crop_base64: string;
}

export interface BenchmarkMetrics {
  preprocess_time_ms: number;
  inference_time_ms: number;
  postprocess_time_ms: number;
  ocr_time_ms: number;
  total_time_ms: number;
  fps: number;
}

export interface ALPRResponse {
  success: boolean;
  total_plates_found: number;
  plates: PlateResult[];
  benchmark: BenchmarkMetrics;
}