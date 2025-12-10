import { useState } from 'react';
import { Search, AlertTriangle, CheckCircle, Info, Loader2 } from 'lucide-react';
import { dgaApi } from '../services/api';
import type { PredictionResult, DetailedPrediction } from '../types';

function ResultCard({ result, detailed }: { result: PredictionResult | DetailedPrediction; detailed?: boolean }) {
  const isDetailed = detailed && 'features' in result;

  return (
    <div className={`card border-2 ${result.is_dga ? 'border-danger-300 bg-danger-50' : 'border-success-300 bg-success-50'}`}>
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center">
          {result.is_dga ? (
            <AlertTriangle className="h-8 w-8 text-danger-500 mr-3" />
          ) : (
            <CheckCircle className="h-8 w-8 text-success-500 mr-3" />
          )}
          <div>
            <h3 className="text-xl font-bold text-gray-900">{result.domain}</h3>
            <p className={`text-sm ${result.is_dga ? 'text-danger-600' : 'text-success-600'}`}>
              {result.is_dga ? 'Potentially Malicious (DGA)' : 'Likely Legitimate'}
            </p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-3xl font-bold text-gray-900">{(result.confidence * 100).toFixed(1)}%</p>
          <p className="text-sm text-gray-500">Confidence</p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="bg-white rounded-lg p-3">
          <p className="text-xs text-gray-500">DGA Probability</p>
          <p className="text-lg font-semibold text-danger-600">{(result.dga_probability * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-white rounded-lg p-3">
          <p className="text-xs text-gray-500">Legit Probability</p>
          <p className="text-lg font-semibold text-success-600">{(result.legit_probability * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-white rounded-lg p-3">
          <p className="text-xs text-gray-500">Model Used</p>
          <p className="text-lg font-semibold text-gray-700">{result.model_used}</p>
        </div>
        <div className="bg-white rounded-lg p-3">
          <p className="text-xs text-gray-500">Analyzed At</p>
          <p className="text-sm font-semibold text-gray-700">{new Date(result.timestamp).toLocaleTimeString()}</p>
        </div>
      </div>

      {isDetailed && (result as DetailedPrediction).features && (
        <div className="mt-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Feature Analysis</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {Object.entries((result as DetailedPrediction).features).map(([key, value]) => (
              <div key={key} className="bg-white rounded p-2">
                <p className="text-xs text-gray-500 truncate" title={key}>{key.replace(/_/g, ' ')}</p>
                <p className="font-mono text-sm">{typeof value === 'number' ? value.toFixed(4) : value}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

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
    } catch (err) {
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
    } catch (err) {
      setError('Failed to analyze domains. Is the API running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Domain Scanner</h1>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setMode('single')}
            className={`px-4 py-2 rounded-lg font-medium ${
              mode === 'single' ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-700'
            }`}
          >
            Single
          </button>
          <button
            onClick={() => setMode('batch')}
            className={`px-4 py-2 rounded-lg font-medium ${
              mode === 'batch' ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-700'
            }`}
          >
            Batch
          </button>
        </div>
      </div>

      <div className="card">
        <div className="flex items-center space-x-4 mb-4">
          <label className="flex items-center">
            <span className="text-sm text-gray-600 mr-2">Model:</span>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="input w-32"
            >
              <option value="auto">Auto</option>
              <option value="rf">Random Forest</option>
              <option value="lstm">LSTM</option>
            </select>
          </label>

          {mode === 'single' && (
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={detailed}
                onChange={(e) => setDetailed(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm text-gray-600">Show features (RF only)</span>
            </label>
          )}
        </div>

        {mode === 'single' ? (
          <div className="flex space-x-4">
            <input
              type="text"
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSingleScan()}
              placeholder="Enter domain (e.g., google.com or suspicious123abc.net)"
              className="input flex-1"
            />
            <button
              onClick={handleSingleScan}
              disabled={loading || !domain.trim()}
              className="btn btn-primary flex items-center"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <>
                  <Search className="h-5 w-5 mr-2" />
                  Scan
                </>
              )}
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <textarea
              value={batchInput}
              onChange={(e) => setBatchInput(e.target.value)}
              placeholder="Enter domains (one per line)&#10;google.com&#10;suspicious123abc.net&#10;facebook.com"
              className="input h-40 font-mono"
            />
            <button
              onClick={handleBatchScan}
              disabled={loading || !batchInput.trim()}
              className="btn btn-primary flex items-center"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <>
                  <Search className="h-5 w-5 mr-2" />
                  Scan All
                </>
              )}
            </button>
          </div>
        )}

        <div className="mt-4 flex items-start space-x-2 text-sm text-gray-500">
          <Info className="h-4 w-4 mt-0.5 flex-shrink-0" />
          <p>
            Enter a domain name or URL. The system will extract the domain and analyze it for DGA patterns.
            High entropy, random characters, and unusual patterns indicate potential DGA activity.
          </p>
        </div>
      </div>

      {error && (
        <div className="card bg-danger-50 border border-danger-200">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-danger-500 mr-2" />
            <p className="text-danger-700">{error}</p>
          </div>
        </div>
      )}

      {/* Single Result */}
      {result && mode === 'single' && (
        <ResultCard result={result} detailed={detailed} />
      )}

      {/* Batch Results */}
      {batchResults && mode === 'batch' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">
              Results ({batchResults.length} domains)
            </h3>
            <div className="flex items-center space-x-4 text-sm">
              <span className="flex items-center text-danger-600">
                <AlertTriangle className="h-4 w-4 mr-1" />
                {batchResults.filter(r => r.is_dga).length} DGA
              </span>
              <span className="flex items-center text-success-600">
                <CheckCircle className="h-4 w-4 mr-1" />
                {batchResults.filter(r => !r.is_dga).length} Legitimate
              </span>
            </div>
          </div>

          <div className="space-y-3">
            {batchResults.map((res, idx) => (
              <div
                key={idx}
                className={`card p-4 flex items-center justify-between ${
                  res.is_dga ? 'border-l-4 border-danger-500' : 'border-l-4 border-success-500'
                }`}
              >
                <div className="flex items-center">
                  {res.is_dga ? (
                    <AlertTriangle className="h-5 w-5 text-danger-500 mr-3" />
                  ) : (
                    <CheckCircle className="h-5 w-5 text-success-500 mr-3" />
                  )}
                  <span className="font-mono">{res.domain}</span>
                </div>
                <div className="flex items-center space-x-4">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    res.is_dga ? 'bg-danger-100 text-danger-700' : 'bg-success-100 text-success-700'
                  }`}>
                    {(res.confidence * 100).toFixed(1)}% confident
                  </span>
                  <span className="text-sm text-gray-500">{res.model_used}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
