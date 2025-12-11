import { useState, useEffect } from 'react';
import { AlertTriangle, Info } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { dgaApi } from '@/services/api';
import type { ModelInfo, HealthStatus } from '@/types';
import { ModelCard } from './components/ModelCard';
import { SystemStatus } from './components/SystemStatus';
import { ModelComparison } from './components/ModelComparison';

export default function Models() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [modelsData, healthData] = await Promise.all([
          dgaApi.getModels(),
          dgaApi.getHealth()
        ]);
        setModels(modelsData.models);
        setHealth(healthData);
        setError(null);
      } catch {
        setError('Failed to fetch model information. Is the API running?');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="border-destructive bg-destructive/10">
        <CardContent className="pt-6">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-destructive mr-2" />
            <p className="text-destructive">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-foreground">Model Information</h1>

      {health && <SystemStatus health={health} />}

      {models.length === 0 ? (
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-start space-x-3">
              <Info className="h-5 w-5 text-primary mt-0.5" />
              <div>
                <h3 className="font-semibold text-foreground">No Models Loaded</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Train the models first by running:
                </p>
                <pre className="mt-2 bg-muted p-3 rounded text-sm font-mono">
                  python -m src.ml.train
                </pre>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          {models.map((model) => (
            <ModelCard key={model.model_type} model={model} />
          ))}
        </div>
      )}

      <ModelComparison />
    </div>
  );
}
