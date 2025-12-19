import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Activity, TrendingUp, User } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { dgaApi } from '@/services/api';
import { useAuth } from '@/contexts/AuthContext';
import type { Stats } from '@/types';
import { StatCard } from './components/StatCard';
import { RecentDetections } from './components/RecentDetections';
import { TopDGADomains } from './components/TopDGADomains';
import { DetectionDistribution } from './components/DetectionDistribution';
import { HourlyActivity } from './components/HourlyActivity';

interface UserDetection {
  id: number;
  domain: string;
  is_dga: boolean;
  confidence: number;
  source: string | null;
  timestamp: string;
}

interface UserStats {
  detections: UserDetection[];
  total: number;
  dga_count: number;
  legit_count: number;
}

export default function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState<Stats | null>(null);
  const [userStats, setUserStats] = useState<UserStats | null>(null);
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

  useEffect(() => {
    const fetchUserStats = async () => {
      if (user) {
        try {
          const data = await dgaApi.getUserDetections(20);
          setUserStats(data);
        } catch (err) {
          console.error('Failed to fetch user stats:', err);
        }
      } else {
        setUserStats(null);
      }
    };

    fetchUserStats();
  }, [user]);

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
        <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
        <span className="text-sm text-muted-foreground">Last updated: {new Date().toLocaleTimeString()}</span>
      </motion.div>

      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.15 }}
      >
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
      </motion.div>

      <motion.div
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
      >
        <DetectionDistribution dgaCount={stats.dga_detected} legitCount={stats.legit_detected} />
        <HourlyActivity hourlyStats={stats.hourly_stats} />
      </motion.div>

      <motion.div
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.25 }}
      >
        <RecentDetections detections={stats.recent_detections} />
        <TopDGADomains domains={stats.top_dga_domains} />
      </motion.div>

      {/* Personal History Section - Only shown when logged in */}
      {user && userStats && (
        <motion.div
          className="space-y-6 pt-6 border-t border-border"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.3 }}
        >
          <div className="flex items-center gap-3">
            <User className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold text-foreground">Your Personal History</h2>
            <Badge variant="secondary">{user.email}</Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <StatCard
              title="Your Total Scans"
              value={userStats.total.toLocaleString()}
              icon={Activity}
              variant="primary"
            />
            <StatCard
              title="Your DGA Detected"
              value={userStats.dga_count.toLocaleString()}
              icon={AlertTriangle}
              variant="danger"
            />
            <StatCard
              title="Your Legitimate"
              value={userStats.legit_count.toLocaleString()}
              icon={CheckCircle}
              variant="success"
            />
          </div>

          {userStats.detections.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Your Recent Detections</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {userStats.detections.map((detection) => (
                    <div
                      key={detection.id}
                      className={`flex items-center justify-between p-3 rounded-lg border ${
                        detection.is_dga
                          ? 'bg-red-500/10 border-red-500/30 dark:bg-red-500/20 dark:border-red-500/40'
                          : 'bg-green-500/10 border-green-500/30 dark:bg-green-500/20 dark:border-green-500/40'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        {detection.is_dga ? (
                          <AlertTriangle className="h-4 w-4 text-red-600 dark:text-red-400" />
                        ) : (
                          <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
                        )}
                        <span className="font-mono text-sm">{detection.domain}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <Badge variant={detection.is_dga ? 'destructive' : 'secondary'}>
                          {(detection.confidence * 100).toFixed(0)}%
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {new Date(detection.timestamp).toLocaleString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </motion.div>
      )}
    </motion.div>
  );
}
