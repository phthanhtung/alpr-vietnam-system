import { useState } from 'react';
import type { ALPRResponse } from '../types/detection';
import { detectPlateApi } from '../services/detectionApi';

export const useDetection = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ALPRResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const analyzeImage = async (file: File) => {
    // Ràng buộc bảo mật cơ bản OWASP ở client: kiểm tra file < 5MB
    if (file.size > 5 * 1024 * 1024) {
      setError('Dung lượng tệp vượt quá giới hạn cho phép (tối đa 5MB)');
      return;
    }

    setPreviewUrl(URL.createObjectURL(file));
    setLoading(true);
    setError(null);

    try {
      const data = await detectPlateApi(file);
      setResult(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Không thể kết nối đến máy chủ nhận diện');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
    setPreviewUrl(null);
  };

  return { loading, result, error, previewUrl, analyzeImage, reset };
};