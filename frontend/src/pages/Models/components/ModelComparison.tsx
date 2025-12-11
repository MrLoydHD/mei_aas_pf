import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function ModelComparison() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-blue-500/10 rounded-lg p-4">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">Random Forest</h4>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>+ Fast inference speed</li>
              <li>+ Interpretable features</li>
              <li>+ Works well with tabular features</li>
              <li>- Requires manual feature engineering</li>
            </ul>
          </div>
          <div className="bg-purple-500/10 rounded-lg p-4">
            <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">LSTM (CNN-LSTM)</h4>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>+ Learns features automatically</li>
              <li>+ Better at sequence patterns</li>
              <li>+ Can capture complex relationships</li>
              <li>- Slower inference</li>
              <li>- Less interpretable</li>
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
