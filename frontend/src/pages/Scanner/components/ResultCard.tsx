import { AlertTriangle, CheckCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import type { PredictionResult, DetailedPrediction } from '@/types';

interface ResultCardProps {
  result: PredictionResult | DetailedPrediction;
  detailed?: boolean;
}

export function ResultCard({ result, detailed }: ResultCardProps) {
  const isDetailed = detailed && 'features' in result;

  return (
    <Card className={`border-2 ${result.is_dga ? 'border-destructive bg-destructive/5' : 'border-success-500 bg-success-50'}`}>
      <CardContent className="pt-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center">
            {result.is_dga ? (
              <AlertTriangle className="h-8 w-8 text-destructive mr-3" />
            ) : (
              <CheckCircle className="h-8 w-8 text-success-500 mr-3" />
            )}
            <div>
              <h3 className="text-xl font-bold text-foreground">{result.domain}</h3>
              <p className={`text-sm ${result.is_dga ? 'text-destructive' : 'text-success-600'}`}>
                {result.is_dga ? 'Potentially Malicious (DGA)' : 'Likely Legitimate'}
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold text-foreground">{(result.confidence * 100).toFixed(1)}%</p>
            <p className="text-sm text-muted-foreground">Confidence</p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div className="bg-card rounded-lg p-3 border">
            <p className="text-xs text-muted-foreground">DGA Probability</p>
            <p className="text-lg font-semibold text-destructive">{(result.dga_probability * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-card rounded-lg p-3 border">
            <p className="text-xs text-muted-foreground">Legit Probability</p>
            <p className="text-lg font-semibold text-success-600">{(result.legit_probability * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-card rounded-lg p-3 border">
            <p className="text-xs text-muted-foreground">Model Used</p>
            <p className="text-lg font-semibold text-foreground">{result.model_used}</p>
          </div>
          <div className="bg-card rounded-lg p-3 border">
            <p className="text-xs text-muted-foreground">Analyzed At</p>
            <p className="text-sm font-semibold text-foreground">{new Date(result.timestamp).toLocaleTimeString()}</p>
          </div>
        </div>

        {isDetailed && (result as DetailedPrediction).features && (
          <div className="mt-4">
            <h4 className="text-sm font-semibold text-foreground mb-2">Feature Analysis</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.entries((result as DetailedPrediction).features).map(([key, value]) => (
                <div key={key} className="bg-card rounded p-2 border">
                  <p className="text-xs text-muted-foreground truncate" title={key}>{key.replace(/_/g, ' ')}</p>
                  <p className="font-mono text-sm">{typeof value === 'number' ? value.toFixed(4) : value}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
