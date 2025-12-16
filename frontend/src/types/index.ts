export interface PredictionResult {
  domain: string;
  is_dga: boolean;
  confidence: number;
  dga_probability: number;
  legit_probability: number;
  model_used: string;
  timestamp: string;
}

export interface DetailedPrediction extends PredictionResult {
  features: Record<string, number>;
}

export interface BatchPredictionResult {
  predictions: PredictionResult[];
  total: number;
  dga_count: number;
  legit_count: number;
}

export interface DetectionLog {
  id: number;
  domain: string;
  is_dga: boolean;
  confidence: number;
  source: string | null;
  user_agent: string | null;
  timestamp: string;
}

export interface Stats {
  total_scans: number;
  dga_detected: number;
  legit_detected: number;
  detection_rate: number;
  top_dga_domains: Array<{
    domain: string;
    count: number;
    avg_confidence: number;
  }>;
  recent_detections: DetectionLog[];
  hourly_stats: Record<string, {
    total: number;
    dga: number;
    legit: number;
  }>;
}

export interface ClassificationReportEntry {
  precision: number;
  recall: number;
  'f1-score': number;
  support: number;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  confusion_matrix?: number[][];
  classification_report?: Record<string, ClassificationReportEntry>;
  history?: {
    loss: number[];
    val_loss: number[];
    accuracy: number[];
    val_accuracy: number[];
    auc?: number[];
    val_auc?: number[];
  };
}

export interface ModelInfo {
  model_name: string;
  model_type: string;
  is_loaded: boolean;
  metrics: ModelMetrics | null;
  feature_importance: Record<string, number> | null;
}

export interface HealthStatus {
  status: string;
  models_loaded: Record<string, boolean>;
  version: string;
  uptime_seconds: number;
}
