import axios from 'axios';
import type {
  PredictionResult,
  DetailedPrediction,
  BatchPredictionResult,
  Stats,
  ModelInfo,
  HealthStatus
} from '../types';

// In production (Docker), use /api which nginx proxies to backend
// In development, use localhost:8000 directly or via Vite proxy
const API_BASE = import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD ? '/api' : 'http://localhost:8000');

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Auth types
interface User {
  id: number;
  email: string;
  name: string | null;
  picture: string | null;
}

interface AuthResponse {
  user: User;
  token: string;
  created: boolean;
}

interface UserStats {
  total_checked: number;
  dga_detected: number;
  legit_detected: number;
}

interface UserDetection {
  id: number;
  domain: string;
  is_dga: boolean;
  confidence: number;
  source: string | null;
  timestamp: string;
}

interface UserDetectionsResponse {
  detections: UserDetection[];
  total: number;
  dga_count: number;
  legit_count: number;
}

export const dgaApi = {
  // Set auth token for authenticated requests
  setAuthToken(token: string | null) {
    if (token) {
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      delete api.defaults.headers.common['Authorization'];
    }
  },

  // Auth endpoints
  async googleAuth(credential: string): Promise<AuthResponse> {
    const response = await api.post('/auth/google', { credential });
    return response.data;
  },

  async verifyToken(): Promise<{ valid: boolean; user_id: string }> {
    const response = await api.post('/auth/verify');
    return response.data;
  },

  async getMe(): Promise<User> {
    const response = await api.get('/auth/me');
    return response.data;
  },

  async getUserStats(): Promise<UserStats> {
    const response = await api.get('/auth/stats');
    return response.data;
  },

  async syncStats(stats: UserStats): Promise<UserStats> {
    const response = await api.post('/auth/sync', stats);
    return response.data;
  },

  async getUserDetections(limit: number = 50): Promise<UserDetectionsResponse> {
    const response = await api.get(`/auth/detections?limit=${limit}`);
    return response.data;
  },

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
