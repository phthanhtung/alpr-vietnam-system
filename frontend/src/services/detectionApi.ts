import axios from 'axios';
import type { ALPRResponse } from '../types/detection';

const API_BASE_URL = 'http://localhost:8000/api/v1';

export const detectPlateApi = async (file: File): Promise<ALPRResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post<ALPRResponse>(`${API_BASE_URL}/detect`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};