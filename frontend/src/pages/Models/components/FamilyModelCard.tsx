import { CheckCircle, XCircle, ChevronRight, Brain, TreeDeciduous, Zap, TrendingUp, Sparkles, Languages, Cpu } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { MetricCard } from './MetricCard';
import type { ModelInfo } from '@/types';

interface FamilyModelCardProps {
  model: ModelInfo;
  onViewDetails: () => void;
}

// Get icon based on family classifier type
function getFamilyModelIcon(modelType: string) {
  const icons: Record<string, typeof Cpu> = {
    'family_classifier_rf': TreeDeciduous,
    'family_classifier_lstm': Brain,
    'family_classifier_xgb': Zap,
    'family_classifier_gb': TrendingUp,
    'family_classifier_transformer': Sparkles,
    'family_classifier_distilbert': Languages,
  };
  return icons[modelType] || Cpu;
}

// Get description based on family classifier type
function getFamilyModelDescription(modelType: string) {
  const descriptions: Record<string, string> = {
    'family_classifier_rf': 'Random Forest with handcrafted features',
    'family_classifier_lstm': 'Character-level LSTM deep learning',
    'family_classifier_xgb': 'Extreme gradient boosting classifier',
    'family_classifier_gb': 'Gradient boosting ensemble classifier',
    'family_classifier_transformer': 'Custom transformer with self-attention',
    'family_classifier_distilbert': 'Fine-tuned DistilBERT language model',
  };
  return descriptions[modelType] || 'Family classification model';
}

// Get icon colors based on family classifier type
function getFamilyModelColors(modelType: string) {
  const colors: Record<string, { bg: string; text: string }> = {
    'family_classifier_rf': { bg: 'bg-destructive/10', text: 'text-destructive' },
    'family_classifier_lstm': { bg: 'bg-purple-500/10', text: 'text-purple-500' },
    'family_classifier_xgb': { bg: 'bg-amber-500/10', text: 'text-amber-500' },
    'family_classifier_gb': { bg: 'bg-green-500/10', text: 'text-green-500' },
    'family_classifier_transformer': { bg: 'bg-pink-500/10', text: 'text-pink-500' },
    'family_classifier_distilbert': { bg: 'bg-blue-500/10', text: 'text-blue-500' },
  };
  return colors[modelType] || { bg: 'bg-primary/10', text: 'text-primary' };
}

export function FamilyModelCard({ model, onViewDetails }: FamilyModelCardProps) {
  // Family classifier has different metric names
  const metrics = model.metrics as Record<string, unknown> | null;

  const accuracy = metrics?.accuracy as number | undefined;
  const precisionMacro = metrics?.precision_macro as number | undefined;
  const recallMacro = metrics?.recall_macro as number | undefined;
  const f1Macro = metrics?.f1_macro as number | undefined;
  const numFamilies = metrics?.num_families as number | undefined;

  const Icon = getFamilyModelIcon(model.model_type);
  const description = getFamilyModelDescription(model.model_type);
  const { bg: iconBgColor, text: iconColor } = getFamilyModelColors(model.model_type);

  return (
    <Card className="transition-all hover:shadow-lg hover:border-primary/50">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-center">
            <div className={`p-2 ${iconBgColor} rounded-lg mr-3`}>
              <Icon className={`h-6 w-6 ${iconColor}`} />
            </div>
            <div>
              <CardTitle className="text-lg">{model.model_name}</CardTitle>
              <p className="text-sm text-muted-foreground">
                {description}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {numFamilies && (
              <Badge variant="outline">
                {numFamilies} Families
              </Badge>
            )}
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
        {metrics && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <MetricCard label="Accuracy" value={accuracy ?? 0} color="text-primary" />
              <MetricCard label="Precision" value={precisionMacro ?? 0} color="text-blue-600" />
              <MetricCard label="Recall" value={recallMacro ?? 0} color="text-purple-600" />
              <MetricCard label="F1 Score" value={f1Macro ?? 0} color="text-indigo-600" />
            </div>
            <div className="flex justify-end">
              <Button variant="ghost" onClick={(e) => { e.stopPropagation(); onViewDetails(); }}>
                View detailed analysis
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}

        {!metrics && (
          <div className="text-center py-4 text-muted-foreground">
            <p>Model not trained yet</p>
            <p className="text-sm mt-1">
              Run training with: <code className="bg-muted px-1 rounded">python -m src.ml.train_family --rf-only</code>
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
