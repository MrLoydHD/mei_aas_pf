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

// Family Classification Types
export interface FamilyInfo {
  family: string;
  confidence: number;
  description: string;
  threat_level: 'critical' | 'high' | 'medium' | 'low' | 'unknown';
  first_seen: string;
  malware_type: string;
  alternatives: Array<{
    family: string;
    confidence: number;
  }>;
}

export interface FamilyPredictionResult {
  domain: string;
  is_dga: boolean;
  dga_confidence: number;
  family_info: FamilyInfo | null;
  model_used: string;
  family_model_used: string | null;
  timestamp: string;
}

export interface DGAFamilyMetadata {
  description: string;
  threat_level: string;
  first_seen: string;
  type: string;
}

export interface FamiliesInfo {
  families: Record<string, DGAFamilyMetadata>;
  total_families: number;
  models_loaded: {
    family_rf: boolean;
    family_lstm: boolean;
    family_xgb: boolean;
    family_gb: boolean;
    family_transformer: boolean;
    family_distilbert: boolean;
  };
  // Legacy fields for backward compatibility
  family_rf_loaded: boolean;
  family_lstm_loaded: boolean;
}

// Model type constants
export type BinaryModelType = 'auto' | 'rf' | 'lstm' | 'xgb' | 'gb' | 'transformer' | 'distilbert';
export type FamilyModelType = 'auto' | 'rf' | 'lstm' | 'xgb' | 'gb' | 'transformer' | 'distilbert';

export const BINARY_MODEL_OPTIONS: Array<{ value: BinaryModelType; label: string; description: string }> = [
  { value: 'auto', label: 'Auto (Best Available)', description: 'Automatically select the best available model' },
  { value: 'rf', label: 'Random Forest', description: 'Tree-based ensemble with handcrafted features' },
  { value: 'lstm', label: 'LSTM', description: 'Deep learning with character-level encoding' },
  { value: 'xgb', label: 'XGBoost', description: 'Gradient boosting with handcrafted features' },
  { value: 'gb', label: 'Gradient Boosting', description: 'Sklearn gradient boosting classifier' },
  { value: 'transformer', label: 'Transformer', description: 'Custom character-level transformer' },
  { value: 'distilbert', label: 'DistilBERT', description: 'Fine-tuned DistilBERT model' },
];

export const FAMILY_MODEL_OPTIONS: Array<{ value: FamilyModelType; label: string; description: string }> = [
  { value: 'auto', label: 'Ensemble (All Models)', description: 'Weighted voting from all available models' },
  { value: 'rf', label: 'Random Forest', description: 'Tree-based ensemble classifier' },
  { value: 'lstm', label: 'LSTM', description: 'Deep learning classifier' },
  { value: 'xgb', label: 'XGBoost', description: 'Gradient boosting classifier' },
  { value: 'gb', label: 'Gradient Boosting', description: 'Sklearn gradient boosting' },
  { value: 'transformer', label: 'Transformer', description: 'Custom transformer classifier' },
  { value: 'distilbert', label: 'DistilBERT', description: 'Fine-tuned DistilBERT' },
];

export interface FamilyStats {
  family: string;
  count: number;
  avg_confidence: number;
  threat_level: string;
}

export interface FamilyStatsResponse {
  family_stats: FamilyStats[];
  threat_level_distribution: Record<string, number>;
}
