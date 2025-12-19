import { AlertTriangle, CheckCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { DetectionLog } from '@/types';

interface RecentDetectionsProps {
  detections: DetectionLog[];
}

export function RecentDetections({ detections }: RecentDetectionsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Detections</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {detections.length === 0 ? (
            <p className="text-muted-foreground text-center py-4">No recent detections</p>
          ) : (
            detections.map((detection) => (
              <div
                key={detection.id}
                className={`flex items-center justify-between p-3 rounded-lg border ${
                  detection.is_dga
                    ? 'bg-red-500/10 border-red-500/30 dark:bg-red-500/20 dark:border-red-500/40'
                    : 'bg-green-500/10 border-green-500/30 dark:bg-green-500/20 dark:border-green-500/40'
                }`}
              >
                <div className="flex items-center">
                  {detection.is_dga ? (
                    <AlertTriangle className="h-5 w-5 text-red-600 dark:text-red-400 mr-3" />
                  ) : (
                    <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400 mr-3" />
                  )}
                  <div>
                    <p className="font-medium text-foreground">{detection.domain}</p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(detection.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
                <Badge variant={detection.is_dga ? 'destructive' : 'secondary'}>
                  {(detection.confidence * 100).toFixed(1)}%
                </Badge>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}
