import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { HealthStatus } from '@/types';

interface SystemStatusProps {
  health: HealthStatus;
}

export function SystemStatus({ health }: SystemStatusProps) {
  return (
    <Card className="bg-gradient-to-r from-primary/10 to-primary/5">
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-foreground">System Status</h3>
            <p className="text-sm text-muted-foreground">Version: {health.version}</p>
          </div>
          <div className="text-right">
            <Badge variant={health.status === 'healthy' ? 'default' : 'destructive'} className="text-lg px-4 py-1">
              {health.status.toUpperCase()}
            </Badge>
            <p className="text-sm text-muted-foreground mt-1">
              Uptime: {Math.floor(health.uptime_seconds / 60)} minutes
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
