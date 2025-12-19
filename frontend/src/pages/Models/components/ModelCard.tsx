import { Cpu, CheckCircle, XCircle, ChevronRight, Brain, TreeDeciduous, Zap, TrendingUp, Sparkles, Languages } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { MetricCard } from './MetricCard';
import type { ModelInfo } from '@/types';

interface ModelCardProps {
  model: ModelInfo;
  onViewDetails: () => void;
}

// Get icon based on model type - scalable for new models
function getModelIcon(modelType: string) {
  const icons: Record<string, typeof Cpu> = {
    'random_forest': TreeDeciduous,
    'lstm': Brain,
    'xgboost': Zap,
    'gradient_boosting': TrendingUp,
    'transformer': Sparkles,
    'distilbert': Languages,
    'cnn': Cpu,
  };
  return icons[modelType] || Cpu;
}

// Get description based on model type - scalable for new models
function getModelDescription(modelType: string) {
  const descriptions: Record<string, string> = {
    'random_forest': 'Handcrafted features with ensemble decision trees',
    'lstm': 'Character-level deep learning with CNN-LSTM architecture',
    'xgboost': 'Extreme gradient boosting with handcrafted features',
    'gradient_boosting': 'Sklearn gradient boosting ensemble classifier',
    'transformer': 'Custom character-level transformer with self-attention',
    'distilbert': 'Fine-tuned DistilBERT pre-trained language model',
    'cnn': 'Convolutional neural network for pattern detection',
  };
  return descriptions[modelType] || 'Machine learning model for DGA detection';
}

export function ModelCard({ model, onViewDetails }: ModelCardProps) {
  const Icon = getModelIcon(model.model_type);
  const description = getModelDescription(model.model_type);

  return (
    <Card
      className="transition-all hover:shadow-lg hover:border-primary/50"
    >
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-center">
            <div className="p-2 bg-primary/10 rounded-lg mr-3">
              <Icon className="h-6 w-6 text-primary" />
            </div>
            <div>
              <CardTitle className="text-lg">{model.model_name}</CardTitle>
              <p className="text-sm text-muted-foreground">{description}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={model.is_loaded ? 'default' : 'destructive'}>
              {model.is_loaded ? (
                <>
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Loaded
                </>
              ) : (
                <>
                  <XCircle className="h-3 w-3 mr-1" />
                  Not Loaded
                </>
              )}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {model.metrics && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <MetricCard label="Accuracy" value={model.metrics.accuracy} color="text-primary" />
              <MetricCard label="Precision" value={model.metrics.precision} color="text-blue-600" />
              <MetricCard label="Recall" value={model.metrics.recall} color="text-purple-600" />
              <MetricCard label="F1 Score" value={model.metrics.f1} color="text-indigo-600" />
              <MetricCard label="ROC-AUC" value={model.metrics.roc_auc} color="text-green-600" />
            </div>
            <div className="flex justify-end">
              <Button variant="ghost" onClick={(e) => { e.stopPropagation(); onViewDetails(); }}>
                View detailed analysis
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}

        {!model.metrics && (
          <div className="text-center py-4 text-muted-foreground">
            <p>Model not trained yet</p>
            <p className="text-sm mt-1">Run training to see metrics</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
