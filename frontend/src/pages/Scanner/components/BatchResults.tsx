import { AlertTriangle, CheckCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { PredictionResult } from '@/types';

interface BatchResultsProps {
  results: PredictionResult[];
}

export function BatchResults({ results }: BatchResultsProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-foreground">
          Results ({results.length} domains)
        </h3>
        <div className="flex items-center space-x-4 text-sm">
          <span className="flex items-center text-destructive">
            <AlertTriangle className="h-4 w-4 mr-1" />
            {results.filter(r => r.is_dga).length} DGA
          </span>
          <span className="flex items-center text-success-600">
            <CheckCircle className="h-4 w-4 mr-1" />
            {results.filter(r => !r.is_dga).length} Legitimate
          </span>
        </div>
      </div>

      <div className="space-y-3">
        {results.map((res, idx) => (
          <Card
            key={idx}
            className={`${res.is_dga ? 'border-l-4 border-l-destructive' : 'border-l-4 border-l-success-500'}`}
          >
            <CardContent className="py-4 flex items-center justify-between">
              <div className="flex items-center">
                {res.is_dga ? (
                  <AlertTriangle className="h-5 w-5 text-destructive mr-3" />
                ) : (
                  <CheckCircle className="h-5 w-5 text-success-500 mr-3" />
                )}
                <span className="font-mono">{res.domain}</span>
              </div>
              <div className="flex items-center space-x-4">
                <Badge variant={res.is_dga ? 'destructive' : 'secondary'}>
                  {(res.confidence * 100).toFixed(1)}% confident
                </Badge>
                <span className="text-sm text-muted-foreground">{res.model_used}</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
