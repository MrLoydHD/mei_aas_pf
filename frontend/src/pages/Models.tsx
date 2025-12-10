import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Cpu, CheckCircle, XCircle, AlertTriangle, Info } from 'lucide-react';
import { dgaApi } from '../services/api';
import type { ModelInfo, HealthStatus } from '../types';

function MetricCard({ label, value, color }: { label: string; value: number; color: string }) {
  const percentage = value * 100;

  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-gray-600">{label}</span>
        <span className={`text-lg font-bold ${color}`}>{percentage.toFixed(2)}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full ${color.replace('text-', 'bg-')}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function ModelCard({ model }: { model: ModelInfo }) {
  const [showFeatures, setShowFeatures] = useState(false);

  return (
    <div className="card">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center">
          <Cpu className="h-8 w-8 text-primary-600 mr-3" />
          <div>
            <h3 className="text-lg font-bold text-gray-900">{model.model_name}</h3>
            <p className="text-sm text-gray-500">Type: {model.model_type}</p>
          </div>
        </div>
        <span
          className={`flex items-center px-3 py-1 rounded-full text-sm font-medium ${
            model.is_loaded ? 'bg-success-100 text-success-700' : 'bg-danger-100 text-danger-700'
          }`}
        >
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
        </span>
      </div>

      {model.metrics && (
        <div className="space-y-4">
          <h4 className="text-sm font-semibold text-gray-700">Performance Metrics</h4>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <MetricCard label="Accuracy" value={model.metrics.accuracy} color="text-primary-600" />
            <MetricCard label="Precision" value={model.metrics.precision} color="text-blue-600" />
            <MetricCard label="Recall" value={model.metrics.recall} color="text-purple-600" />
            <MetricCard label="F1 Score" value={model.metrics.f1} color="text-indigo-600" />
            <MetricCard label="ROC-AUC" value={model.metrics.roc_auc} color="text-green-600" />
          </div>
        </div>
      )}

      {model.feature_importance && (
        <div className="mt-6">
          <button
            onClick={() => setShowFeatures(!showFeatures)}
            className="text-sm text-primary-600 hover:text-primary-700 font-medium"
          >
            {showFeatures ? 'Hide' : 'Show'} Feature Importance
          </button>

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
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 'auto']} />
                  <YAxis dataKey="name" type="category" fontSize={11} width={110} />
                  <Tooltip formatter={(value: number) => value.toFixed(4)} />
                  <Bar dataKey="value" fill="#0ea5e9" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

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
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card bg-danger-50 border border-danger-200">
        <div className="flex items-center">
          <AlertTriangle className="h-5 w-5 text-danger-500 mr-2" />
          <p className="text-danger-700">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Model Information</h1>

      {/* System Status */}
      {health && (
        <div className="card bg-gradient-to-r from-primary-50 to-primary-100">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-primary-900">System Status</h3>
              <p className="text-sm text-primary-700">Version: {health.version}</p>
            </div>
            <div className="text-right">
              <p className={`text-lg font-bold ${health.status === 'healthy' ? 'text-success-600' : 'text-danger-600'}`}>
                {health.status.toUpperCase()}
              </p>
              <p className="text-sm text-primary-700">
                Uptime: {Math.floor(health.uptime_seconds / 60)} minutes
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Model Info */}
      {models.length === 0 ? (
        <div className="card">
          <div className="flex items-start space-x-3">
            <Info className="h-5 w-5 text-primary-500 mt-0.5" />
            <div>
              <h3 className="font-semibold text-gray-900">No Models Loaded</h3>
              <p className="text-sm text-gray-600 mt-1">
                Train the models first by running:
              </p>
              <pre className="mt-2 bg-gray-100 p-3 rounded text-sm font-mono">
                python -m src.ml.train
              </pre>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {models.map((model) => (
            <ModelCard key={model.model_type} model={model} />
          ))}
        </div>
      )}

      {/* Model Comparison Info */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Comparison</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2">Random Forest</h4>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>+ Fast inference speed</li>
              <li>+ Interpretable features</li>
              <li>+ Works well with tabular features</li>
              <li>- Requires manual feature engineering</li>
            </ul>
          </div>
          <div className="bg-purple-50 rounded-lg p-4">
            <h4 className="font-semibold text-purple-900 mb-2">LSTM (CNN-LSTM)</h4>
            <ul className="text-sm text-purple-800 space-y-1">
              <li>+ Learns features automatically</li>
              <li>+ Better at sequence patterns</li>
              <li>+ Can capture complex relationships</li>
              <li>- Slower inference</li>
              <li>- Less interpretable</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
