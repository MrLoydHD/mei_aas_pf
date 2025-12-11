import { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, Activity, TrendingUp } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { dgaApi } from '@/services/api';
import type { Stats } from '@/types';
import { StatCard } from './components/StatCard';
import { RecentDetections } from './components/RecentDetections';
import { TopDGADomains } from './components/TopDGADomains';
import { DetectionDistribution } from './components/DetectionDistribution';
import { HourlyActivity } from './components/HourlyActivity';

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await dgaApi.getStats(24);
        setStats(data);
        setError(null);
      } catch {
        setError('Failed to fetch statistics. Is the API running?');
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
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

  if (!stats) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
        <span className="text-sm text-muted-foreground">Last updated: {new Date().toLocaleTimeString()}</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Scans"
          value={stats.total_scans.toLocaleString()}
          icon={Activity}
          variant="primary"
        />
        <StatCard
          title="DGA Detected"
          value={stats.dga_detected.toLocaleString()}
          icon={AlertTriangle}
          variant="danger"
          subtitle={`${(stats.detection_rate * 100).toFixed(1)}% of total`}
        />
        <StatCard
          title="Legitimate"
          value={stats.legit_detected.toLocaleString()}
          icon={CheckCircle}
          variant="success"
        />
        <StatCard
          title="Detection Rate"
          value={`${(stats.detection_rate * 100).toFixed(1)}%`}
          icon={TrendingUp}
          variant="primary"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DetectionDistribution dgaCount={stats.dga_detected} legitCount={stats.legit_detected} />
        <HourlyActivity hourlyStats={stats.hourly_stats} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RecentDetections detections={stats.recent_detections} />
        <TopDGADomains domains={stats.top_dga_domains} />
      </div>
    </div>
  );
}
