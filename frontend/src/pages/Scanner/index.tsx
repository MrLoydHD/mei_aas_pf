import { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { dgaApi } from '@/services/api';
import type { PredictionResult, DetailedPrediction } from '@/types';
import { ScannerForm } from './components/ScannerForm';
import { ResultCard } from './components/ResultCard';
import { BatchResults } from './components/BatchResults';

export default function Scanner() {
  const [domain, setDomain] = useState('');
  const [batchInput, setBatchInput] = useState('');
  const [result, setResult] = useState<PredictionResult | DetailedPrediction | null>(null);
  const [batchResults, setBatchResults] = useState<PredictionResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [detailed, setDetailed] = useState(false);
  const [modelType, setModelType] = useState('auto');

  const handleSingleScan = async () => {
    if (!domain.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = detailed
        ? await dgaApi.predictDetailed(domain)
        : await dgaApi.predict(domain, modelType);
      setResult(data);
    } catch {
      setError('Failed to analyze domain. Is the API running?');
    } finally {
      setLoading(false);
    }
  };

  const handleBatchScan = async () => {
    const domains = batchInput.split('\n').map(d => d.trim()).filter(d => d);
    if (domains.length === 0) return;

    setLoading(true);
    setError(null);
    setBatchResults(null);

    try {
      const data = await dgaApi.predictBatch(domains, modelType);
      setBatchResults(data.predictions);
    } catch {
      setError('Failed to analyze domains. Is the API running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <motion.div
        className="flex items-center justify-between"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        <h1 className="text-2xl font-bold text-foreground">Domain Scanner</h1>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.15 }}
      >
        <ScannerForm
        mode={mode}
        onModeChange={setMode}
        domain={domain}
        onDomainChange={setDomain}
        batchInput={batchInput}
        onBatchInputChange={setBatchInput}
        modelType={modelType}
        onModelTypeChange={setModelType}
        detailed={detailed}
        onDetailedChange={setDetailed}
        loading={loading}
        onSingleScan={handleSingleScan}
        onBatchScan={handleBatchScan}
      />
      </motion.div>

      {error && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          <Card className="border-destructive bg-destructive/10">
            <CardContent className="pt-6">
              <div className="flex items-center">
                <AlertTriangle className="h-5 w-5 text-destructive mr-2" />
                <p className="text-destructive">{error}</p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {result && mode === 'single' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <ResultCard result={result} detailed={detailed} />
        </motion.div>
      )}

      {batchResults && mode === 'batch' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <BatchResults results={batchResults} />
        </motion.div>
      )}
    </motion.div>
  );
}
