import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Cpu, CheckCircle, XCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { MetricCard } from './MetricCard';
import type { ModelInfo } from '@/types';

interface ModelCardProps {
  model: ModelInfo;
}

export function ModelCard({ model }: ModelCardProps) {
  const [showFeatures, setShowFeatures] = useState(false);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-center">
            <Cpu className="h-8 w-8 text-primary mr-3" />
            <div>
              <CardTitle>{model.model_name}</CardTitle>
              <p className="text-sm text-muted-foreground">Type: {model.model_type}</p>
            </div>
          </div>
          <Badge variant={model.is_loaded ? 'default' : 'destructive'}>
            {model.is_loaded ? (
              <>
                <CheckCircle className="h-4 w-4 mr-1" />
                Loaded
              </>
            ) : (
              <>
                <XCircle className="h-4 w-4 mr-1" />
                Not Loaded
              </>
            )}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        {model.metrics && (
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-foreground">Performance Metrics</h4>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <MetricCard label="Accuracy" value={model.metrics.accuracy} color="text-primary" />
              <MetricCard label="Precision" value={model.metrics.precision} color="text-blue-600" />
              <MetricCard label="Recall" value={model.metrics.recall} color="text-purple-600" />
              <MetricCard label="F1 Score" value={model.metrics.f1} color="text-indigo-600" />
              <MetricCard label="ROC-AUC" value={model.metrics.roc_auc} color="text-green-600" />
            </div>
          </div>
        )}

        {model.feature_importance && (
          <div className="mt-6">
            <Button
              variant="link"
              onClick={() => setShowFeatures(!showFeatures)}
              className="p-0 h-auto"
            >
              {showFeatures ? 'Hide' : 'Show'} Feature Importance
            </Button>

            {showFeatures && (
              <div className="mt-4">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={Object.entries(model.feature_importance)
                      .sort(([, a], [, b]) => b - a)
                      .map(([name, value]) => ({ name: name.replace(/_/g, ' '), value }))}
                    layout="vertical"
                    margin={{ left: 120 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                    <XAxis type="number" domain={[0, 'auto']} className="fill-muted-foreground" />
                    <YAxis dataKey="name" type="category" fontSize={11} width={110} className="fill-muted-foreground" />
                    <Tooltip formatter={(value: number) => value.toFixed(4)} />
                    <Bar dataKey="value" fill="var(--primary)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
