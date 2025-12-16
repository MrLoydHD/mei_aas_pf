import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, LineChart, Line, Area, AreaChart
} from 'recharts';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import type { ModelInfo } from '@/types';

interface ModelDetailDialogProps {
  model: ModelInfo | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

// Chart colors that work with both light and dark themes
const CHART_COLORS = {
  primary: '#6366f1',      // Indigo - main color
  success: '#22c55e',      // Green - correct predictions
  danger: '#ef4444',       // Red - errors/destructive
  purple: '#8b5cf6',       // Purple - secondary metric
  blue: '#3b82f6',         // Blue - tertiary metric
  amber: '#f59e0b',        // Amber - warning/highlight
  cyan: '#06b6d4',         // Cyan - additional metric
  text: '#94a3b8',         // Slate - axis text
  grid: '#334155',         // Slate dark - grid lines
  background: '#1e293b',   // Card background for tooltips
  border: '#475569',       // Border color
};

const METRIC_COLORS = [
  CHART_COLORS.primary,
  CHART_COLORS.blue,
  CHART_COLORS.purple,
  CHART_COLORS.cyan,
  CHART_COLORS.success
];

export function ModelDetailDialog({ model, open, onOpenChange }: ModelDetailDialogProps) {
  if (!model) return null;

  const metrics = model.metrics;
  const confusionMatrix = metrics?.confusion_matrix;
  const trainingHistory = metrics?.history;
  const isLSTM = model.model_type === 'lstm';
  const isRandomForest = model.model_type === 'random_forest';

  // Prepare metrics data for radar chart
  const radarData = metrics ? [
    { metric: 'Accuracy', value: metrics.accuracy * 100, fullMark: 100 },
    { metric: 'Precision', value: metrics.precision * 100, fullMark: 100 },
    { metric: 'Recall', value: metrics.recall * 100, fullMark: 100 },
    { metric: 'F1 Score', value: metrics.f1 * 100, fullMark: 100 },
    { metric: 'ROC-AUC', value: metrics.roc_auc * 100, fullMark: 100 },
  ] : [];

  // Prepare metrics data for bar chart
  const metricsBarData = metrics ? [
    { name: 'Accuracy', value: metrics.accuracy * 100 },
    { name: 'Precision', value: metrics.precision * 100 },
    { name: 'Recall', value: metrics.recall * 100 },
    { name: 'F1 Score', value: metrics.f1 * 100 },
    { name: 'ROC-AUC', value: metrics.roc_auc * 100 },
  ] : [];

  // Prepare confusion matrix data
  const confusionData = confusionMatrix ? [
    { name: 'True Negative', value: confusionMatrix[0][0], type: 'correct' },
    { name: 'False Positive', value: confusionMatrix[0][1], type: 'error' },
    { name: 'False Negative', value: confusionMatrix[1][0], type: 'error' },
    { name: 'True Positive', value: confusionMatrix[1][1], type: 'correct' },
  ] : [];

  // Prepare prediction distribution (TP, TN, FP, FN)
  const predictionDistribution = confusionMatrix ? [
    { name: 'Correct', value: confusionMatrix[0][0] + confusionMatrix[1][1] },
    { name: 'Errors', value: confusionMatrix[0][1] + confusionMatrix[1][0] },
  ] : [];

  // Feature importance data (top 15) - Random Forest only
  const featureData = model.feature_importance
    ? Object.entries(model.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 15)
        .map(([name, value]) => ({
          name: name.replace(/_/g, ' ').slice(0, 20),
          value: value,
          fullName: name.replace(/_/g, ' '),
        }))
    : [];

  // Training history data - LSTM only
  const trainingData = trainingHistory
    ? (trainingHistory.loss || []).map((loss: number, index: number) => ({
        epoch: index + 1,
        loss: loss,
        val_loss: trainingHistory.val_loss?.[index] || 0,
        accuracy: (trainingHistory.accuracy?.[index] || 0) * 100,
        val_accuracy: (trainingHistory.val_accuracy?.[index] || 0) * 100,
      }))
    : [];

  // Determine which optional tabs should show
  const showFeaturesTab = isRandomForest && model.feature_importance;
  const showTrainingTab = isLSTM && trainingHistory;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            {model.model_name}
            <Badge variant={model.is_loaded ? 'default' : 'destructive'}>
              {model.is_loaded ? 'Loaded' : 'Not Loaded'}
            </Badge>
          </DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList className={`grid w-full ${showFeaturesTab || showTrainingTab ? 'grid-cols-4' : 'grid-cols-3'}`}>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="confusion">Confusion Matrix</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            {isRandomForest && model.feature_importance && (
              <TabsTrigger value="features">Features</TabsTrigger>
            )}
            {isLSTM && trainingHistory && (
              <TabsTrigger value="training">Training</TabsTrigger>
            )}
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Radar Chart */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Performance Overview</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke={CHART_COLORS.grid} />
                      <PolarAngleAxis
                        dataKey="metric"
                        tick={{ fill: CHART_COLORS.text, fontSize: 12 }}
                      />
                      <PolarRadiusAxis
                        angle={90}
                        domain={[0, 100]}
                        tick={{ fill: CHART_COLORS.text, fontSize: 10 }}
                      />
                      <Radar
                        name="Score"
                        dataKey="value"
                        stroke={CHART_COLORS.primary}
                        fill={CHART_COLORS.primary}
                        fillOpacity={0.5}
                      />
                      <Tooltip
                        formatter={(value: number) => [`${value.toFixed(2)}%`, 'Score']}
                        contentStyle={{
                          backgroundColor: CHART_COLORS.background,
                          border: `1px solid ${CHART_COLORS.border}`,
                          borderRadius: '8px',
                        }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Prediction Distribution Pie */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Prediction Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={predictionDistribution}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                      >
                        <Cell fill={CHART_COLORS.success} />
                        <Cell fill={CHART_COLORS.danger} />
                      </Pie>
                      <Tooltip
                        formatter={(value: number) => [value.toLocaleString(), 'Count']}
                        contentStyle={{
                          backgroundColor: CHART_COLORS.background,
                          border: `1px solid ${CHART_COLORS.border}`,
                          borderRadius: '8px',
                        }}
                      />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            {/* Summary Stats */}
            {metrics && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Model Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
                    <div>
                      <p className="text-2xl font-bold text-primary">{(metrics.accuracy * 100).toFixed(2)}%</p>
                      <p className="text-sm text-muted-foreground">Accuracy</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-blue-500">{(metrics.precision * 100).toFixed(2)}%</p>
                      <p className="text-sm text-muted-foreground">Precision</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-purple-500">{(metrics.recall * 100).toFixed(2)}%</p>
                      <p className="text-sm text-muted-foreground">Recall</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-indigo-500">{(metrics.f1 * 100).toFixed(2)}%</p>
                      <p className="text-sm text-muted-foreground">F1 Score</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-green-500">{(metrics.roc_auc * 100).toFixed(2)}%</p>
                      <p className="text-sm text-muted-foreground">ROC-AUC</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Confusion Matrix Tab */}
          <TabsContent value="confusion" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Confusion Matrix Grid */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Confusion Matrix</CardTitle>
                </CardHeader>
                <CardContent>
                  {confusionMatrix && (
                    <div className="flex flex-col items-center">
                      <div className="mb-2 text-sm text-muted-foreground">Predicted</div>
                      <div className="flex">
                        <div className="flex flex-col justify-center mr-2">
                          <div className="text-sm text-muted-foreground transform -rotate-90 origin-center whitespace-nowrap">
                            Actual
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="text-center text-xs text-muted-foreground mb-1">Legit</div>
                          <div className="text-center text-xs text-muted-foreground mb-1">DGA</div>
                          <div
                            className="w-24 h-24 flex flex-col items-center justify-center rounded-lg bg-green-500/20 border-2 border-green-500 cursor-pointer hover:bg-green-500/30 transition-colors"
                            title={`True Negative: ${confusionMatrix[0][0]}`}
                          >
                            <span className="text-2xl font-bold">{confusionMatrix[0][0]}</span>
                            <span className="text-xs text-muted-foreground">TN</span>
                          </div>
                          <div
                            className="w-24 h-24 flex flex-col items-center justify-center rounded-lg bg-red-500/20 border-2 border-red-500 cursor-pointer hover:bg-red-500/30 transition-colors"
                            title={`False Positive: ${confusionMatrix[0][1]}`}
                          >
                            <span className="text-2xl font-bold">{confusionMatrix[0][1]}</span>
                            <span className="text-xs text-muted-foreground">FP</span>
                          </div>
                          <div
                            className="w-24 h-24 flex flex-col items-center justify-center rounded-lg bg-red-500/20 border-2 border-red-500 cursor-pointer hover:bg-red-500/30 transition-colors"
                            title={`False Negative: ${confusionMatrix[1][0]}`}
                          >
                            <span className="text-2xl font-bold">{confusionMatrix[1][0]}</span>
                            <span className="text-xs text-muted-foreground">FN</span>
                          </div>
                          <div
                            className="w-24 h-24 flex flex-col items-center justify-center rounded-lg bg-green-500/20 border-2 border-green-500 cursor-pointer hover:bg-green-500/30 transition-colors"
                            title={`True Positive: ${confusionMatrix[1][1]}`}
                          >
                            <span className="text-2xl font-bold">{confusionMatrix[1][1]}</span>
                            <span className="text-xs text-muted-foreground">TP</span>
                          </div>
                        </div>
                      </div>
                      <div className="mt-4 flex gap-4 text-xs">
                        <div className="flex items-center gap-1">
                          <div className="w-3 h-3 rounded bg-green-500"></div>
                          <span>Correct</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <div className="w-3 h-3 rounded bg-red-500"></div>
                          <span>Error</span>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Confusion Bar Chart */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Confusion Breakdown</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={confusionData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                      <XAxis type="number" tick={{ fill: CHART_COLORS.text }} />
                      <YAxis
                        dataKey="name"
                        type="category"
                        width={100}
                        tick={{ fill: CHART_COLORS.text, fontSize: 12 }}
                      />
                      <Tooltip
                        formatter={(value: number) => [value.toLocaleString(), 'Count']}
                        contentStyle={{
                          backgroundColor: CHART_COLORS.background,
                          border: `1px solid ${CHART_COLORS.border}`,
                          borderRadius: '8px',
                        }}
                      />
                      <Bar
                        dataKey="value"
                        fill={CHART_COLORS.primary}
                        radius={[0, 4, 4, 0]}
                      >
                        {confusionData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.type === 'correct' ? CHART_COLORS.success : CHART_COLORS.danger}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Metrics Tab */}
          <TabsContent value="metrics" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Performance Metrics Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={metricsBarData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: CHART_COLORS.text }}
                    />
                    <YAxis
                      domain={[0, 100]}
                      tick={{ fill: CHART_COLORS.text }}
                      tickFormatter={(value) => `${value}%`}
                    />
                    <Tooltip
                      formatter={(value: number) => [`${value.toFixed(2)}%`, 'Score']}
                      contentStyle={{
                        backgroundColor: CHART_COLORS.background,
                        border: `1px solid ${CHART_COLORS.border}`,
                        borderRadius: '8px',
                      }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {metricsBarData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={METRIC_COLORS[index % METRIC_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Classification Report */}
            {metrics?.classification_report && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Classification Report</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2">Class</th>
                          <th className="text-right py-2">Precision</th>
                          <th className="text-right py-2">Recall</th>
                          <th className="text-right py-2">F1-Score</th>
                          <th className="text-right py-2">Support</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(metrics.classification_report)
                          .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
                          .map(([className, values]) => (
                            <tr key={className} className="border-b">
                              <td className="py-2">{className === '0' ? 'Legitimate' : 'DGA'}</td>
                              <td className="text-right">{(values.precision * 100).toFixed(2)}%</td>
                              <td className="text-right">{(values.recall * 100).toFixed(2)}%</td>
                              <td className="text-right">{(values['f1-score'] * 100).toFixed(2)}%</td>
                              <td className="text-right">{values.support}</td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Features Tab - Random Forest Only */}
          {isRandomForest && model.feature_importance && (
            <TabsContent value="features" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Top 15 Feature Importance</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={500}>
                    <BarChart data={featureData} layout="vertical" margin={{ left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                      <XAxis
                        type="number"
                        tick={{ fill: CHART_COLORS.text }}
                        tickFormatter={(value) => value.toFixed(3)}
                      />
                      <YAxis
                        dataKey="name"
                        type="category"
                        width={150}
                        tick={{ fill: CHART_COLORS.text, fontSize: 11 }}
                      />
                      <Tooltip
                        formatter={(value: number, _, props) => [
                          value.toFixed(4),
                          props.payload.fullName
                        ]}
                        contentStyle={{
                          backgroundColor: CHART_COLORS.background,
                          border: `1px solid ${CHART_COLORS.border}`,
                          borderRadius: '8px',
                        }}
                      />
                      <Bar
                        dataKey="value"
                        fill={CHART_COLORS.primary}
                        radius={[0, 4, 4, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Feature Categories */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Feature Categories</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="p-3 rounded-lg bg-muted/50">
                      <p className="font-medium">Statistical</p>
                      <p className="text-muted-foreground text-xs">Length, entropy, char distribution</p>
                    </div>
                    <div className="p-3 rounded-lg bg-muted/50">
                      <p className="font-medium">N-gram</p>
                      <p className="text-muted-foreground text-xs">Bigram, trigram frequencies</p>
                    </div>
                    <div className="p-3 rounded-lg bg-muted/50">
                      <p className="font-medium">Linguistic</p>
                      <p className="text-muted-foreground text-xs">Dictionary words, vowel ratios</p>
                    </div>
                    <div className="p-3 rounded-lg bg-muted/50">
                      <p className="font-medium">TF-IDF</p>
                      <p className="text-muted-foreground text-xs">Character pattern analysis</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          )}

          {/* Training Tab - LSTM Only */}
          {isLSTM && trainingHistory && (
            <TabsContent value="training" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Loss Chart */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Training & Validation Loss</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={trainingData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                        <XAxis
                          dataKey="epoch"
                          tick={{ fill: CHART_COLORS.text }}
                          label={{ value: 'Epoch', position: 'bottom', fill: CHART_COLORS.text }}
                        />
                        <YAxis
                          tick={{ fill: CHART_COLORS.text }}
                          label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: CHART_COLORS.text }}
                        />
                        <Tooltip
                          formatter={(value: number) => [value.toFixed(4), '']}
                          contentStyle={{
                            backgroundColor: CHART_COLORS.background,
                            border: `1px solid ${CHART_COLORS.border}`,
                            borderRadius: '8px',
                          }}
                        />
                        <Legend />
                        <Area
                          type="monotone"
                          dataKey="loss"
                          stroke={CHART_COLORS.primary}
                          fill={CHART_COLORS.primary}
                          fillOpacity={0.3}
                          name="Training Loss"
                        />
                        <Area
                          type="monotone"
                          dataKey="val_loss"
                          stroke={CHART_COLORS.danger}
                          fill={CHART_COLORS.danger}
                          fillOpacity={0.3}
                          name="Validation Loss"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                {/* Accuracy Chart */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Training & Validation Accuracy</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={trainingData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                        <XAxis
                          dataKey="epoch"
                          tick={{ fill: CHART_COLORS.text }}
                          label={{ value: 'Epoch', position: 'bottom', fill: CHART_COLORS.text }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tick={{ fill: CHART_COLORS.text }}
                          tickFormatter={(value) => `${value}%`}
                          label={{ value: 'Accuracy', angle: -90, position: 'insideLeft', fill: CHART_COLORS.text }}
                        />
                        <Tooltip
                          formatter={(value: number) => [`${value.toFixed(2)}%`, '']}
                          contentStyle={{
                            backgroundColor: CHART_COLORS.background,
                            border: `1px solid ${CHART_COLORS.border}`,
                            borderRadius: '8px',
                          }}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="accuracy"
                          stroke={CHART_COLORS.primary}
                          strokeWidth={2}
                          dot={false}
                          name="Training Accuracy"
                        />
                        <Line
                          type="monotone"
                          dataKey="val_accuracy"
                          stroke={CHART_COLORS.success}
                          strokeWidth={2}
                          dot={false}
                          name="Validation Accuracy"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>

              {/* LSTM Architecture Info */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Model Architecture</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="p-3 rounded-lg bg-muted/50">
                      <p className="font-medium">Embedding</p>
                      <p className="text-muted-foreground text-xs">Character-level embeddings</p>
                    </div>
                    <div className="p-3 rounded-lg bg-muted/50">
                      <p className="font-medium">Conv1D</p>
                      <p className="text-muted-foreground text-xs">Local pattern detection</p>
                    </div>
                    <div className="p-3 rounded-lg bg-muted/50">
                      <p className="font-medium">Bidirectional LSTM</p>
                      <p className="text-muted-foreground text-xs">Sequential pattern learning</p>
                    </div>
                    <div className="p-3 rounded-lg bg-muted/50">
                      <p className="font-medium">Dense + Dropout</p>
                      <p className="text-muted-foreground text-xs">Classification with regularization</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Training Summary */}
              {trainingData.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Training Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                      <div>
                        <p className="text-2xl font-bold text-primary">{trainingData.length}</p>
                        <p className="text-sm text-muted-foreground">Epochs</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-blue-500">
                          {trainingData[trainingData.length - 1]?.loss.toFixed(4)}
                        </p>
                        <p className="text-sm text-muted-foreground">Final Loss</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-green-500">
                          {trainingData[trainingData.length - 1]?.accuracy.toFixed(1)}%
                        </p>
                        <p className="text-sm text-muted-foreground">Final Train Acc</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-purple-500">
                          {trainingData[trainingData.length - 1]?.val_accuracy.toFixed(1)}%
                        </p>
                        <p className="text-sm text-muted-foreground">Final Val Acc</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
          )}
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
