import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Info, Shield, Cpu } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { dgaApi } from '@/services/api';
import type { ModelInfo, HealthStatus } from '@/types';
import { ModelCard } from './components/ModelCard';
import { FamilyModelCard } from './components/FamilyModelCard';
import { ModelDetailDialog } from './components/ModelDetailDialog';
import { SystemStatus } from './components/SystemStatus';
import { ModelComparison } from './components/ModelComparison';

export default function Models() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  // Separate models into detection (binary) and classification (family)
  const { detectionModels, classificationModels } = useMemo(() => {
    // Binary DGA detection models
    const binaryModelTypes = [
      'random_forest', 'lstm', 'xgboost', 'gradient_boosting', 'transformer', 'distilbert'
    ];
    // Family classification models
    const familyModelTypes = [
      'family_classifier_rf', 'family_classifier_lstm', 'family_classifier_xgb',
      'family_classifier_gb', 'family_classifier_transformer', 'family_classifier_distilbert'
    ];

    const detection = models.filter(m => binaryModelTypes.includes(m.model_type));
    const classification = models.filter(m => familyModelTypes.includes(m.model_type));
    return { detectionModels: detection, classificationModels: classification };
  }, [models]);

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
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <motion.h1
        className="text-2xl font-bold text-foreground"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        Model Information
      </motion.h1>

      {health && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.15 }}
        >
          <SystemStatus health={health} />
        </motion.div>
      )}

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
      >
        <Tabs defaultValue="detection" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="detection" className="flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              Detection Models
              {detectionModels.length > 0 && (
                <span className="ml-1 text-xs bg-primary/20 px-1.5 py-0.5 rounded-full">
                  {detectionModels.length}
                </span>
              )}
            </TabsTrigger>
            <TabsTrigger value="classification" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Family Classification
              {classificationModels.length > 0 && (
                <span className="ml-1 text-xs bg-primary/20 px-1.5 py-0.5 rounded-full">
                  {classificationModels.length}
                </span>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="detection" className="space-y-6">
            {detectionModels.length === 0 ? (
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-start space-x-3">
                    <Info className="h-5 w-5 text-primary mt-0.5" />
                    <div>
                      <h3 className="font-semibold text-foreground">No Detection Models Loaded</h3>
                      <p className="text-sm text-muted-foreground mt-1">
                        Train the binary DGA detection models first:
                      </p>
                      <pre className="mt-2 bg-muted p-3 rounded text-sm font-mono">
                        python -m src.ml.train
                      </pre>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 gap-6">
                {detectionModels.map((model, index) => (
                  <motion.div
                    key={model.model_type}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <ModelCard
                      model={model}
                      onViewDetails={() => {
                        setSelectedModel(model);
                        setDialogOpen(true);
                      }}
                    />
                  </motion.div>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="classification" className="space-y-6">
            {classificationModels.length === 0 ? (
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-start space-x-3">
                    <Info className="h-5 w-5 text-destructive mt-0.5" />
                    <div>
                      <h3 className="font-semibold text-foreground">No Family Classifier Loaded</h3>
                      <p className="text-sm text-muted-foreground mt-1">
                        Train the DGA family classification model:
                      </p>
                      <pre className="mt-2 bg-muted p-3 rounded text-sm font-mono">
                        python -m src.ml.train_family --rf-only
                      </pre>
                      <p className="text-sm text-muted-foreground mt-2">
                        This model classifies DGA domains into specific malware families
                        (e.g., Conficker, CryptoLocker, Emotet).
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 gap-6">
                {classificationModels.map((model, index) => (
                  <motion.div
                    key={model.model_type}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <FamilyModelCard
                      model={model}
                      onViewDetails={() => {
                        setSelectedModel(model);
                        setDialogOpen(true);
                      }}
                    />
                  </motion.div>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.35 }}
      >
        <ModelComparison />
      </motion.div>

      <ModelDetailDialog
        model={selectedModel}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
      />
    </motion.div>
  );
}
