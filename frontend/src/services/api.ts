import axios from 'axios';
import type {
  PredictionResult,
  DetailedPrediction,
  BatchPredictionResult,
  Stats,
  ModelInfo,
  HealthStatus
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const dgaApi = {
  // Health check
  async getHealth(): Promise<HealthStatus> {
    const response = await api.get('/health');
    return response.data;
  },

  // Single domain prediction
  async predict(domain: string, modelType: string = 'auto'): Promise<PredictionResult> {
    const response = await api.post(`/predict?model_type=${modelType}`, { domain });
    return response.data;
  },

  // Detailed prediction with features
  async predictDetailed(domain: string): Promise<DetailedPrediction> {
    const response = await api.post('/predict/detailed', { domain });
    return response.data;
  },

  // Batch prediction
  async predictBatch(domains: string[], modelType: string = 'auto'): Promise<BatchPredictionResult> {
    const response = await api.post(`/predict/batch?model_type=${modelType}`, { domains });
    return response.data;
  },

  // Get statistics
  async getStats(hours: number = 24): Promise<Stats> {
    const response = await api.get(`/stats?hours=${hours}`);
    return response.data;
  },

  // Get model info
  async getModels(): Promise<{ models: ModelInfo[] }> {
    const response = await api.get('/models');
    return response.data;
  },
};

export default dgaApi;
