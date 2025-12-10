import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar
} from 'recharts';
import { Shield, AlertTriangle, CheckCircle, Activity, TrendingUp } from 'lucide-react';
import { dgaApi } from '../services/api';
import type { Stats, DetectionLog } from '../types';

const COLORS = {
  dga: '#ef4444',
  legit: '#22c55e',
  primary: '#0ea5e9'
};

function StatCard({ title, value, icon: Icon, color, subtitle }: {
  title: string;
  value: string | number;
  icon: React.ElementType;
  color: string;
  subtitle?: string;
}) {
  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className={`text-3xl font-bold ${color}`}>{value}</p>
          {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 rounded-full ${color.replace('text-', 'bg-').replace('-600', '-100')}`}>
          <Icon className={`h-6 w-6 ${color}`} />
        </div>
      </div>
    </div>
  );
}

function RecentDetections({ detections }: { detections: DetectionLog[] }) {
  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Detections</h3>
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {detections.length === 0 ? (
          <p className="text-gray-500 text-center py-4">No recent detections</p>
        ) : (
          detections.map((detection) => (
            <div
              key={detection.id}
              className={`flex items-center justify-between p-3 rounded-lg ${
                detection.is_dga ? 'bg-danger-50' : 'bg-success-50'
              }`}
            >
              <div className="flex items-center">
                {detection.is_dga ? (
                  <AlertTriangle className="h-5 w-5 text-danger-500 mr-3" />
                ) : (
                  <CheckCircle className="h-5 w-5 text-success-500 mr-3" />
                )}
                <div>
                  <p className="font-medium text-gray-900">{detection.domain}</p>
                  <p className="text-xs text-gray-500">
                    {new Date(detection.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
              <span
                className={`px-2 py-1 rounded-full text-xs font-medium ${
                  detection.is_dga
                    ? 'bg-danger-100 text-danger-700'
                    : 'bg-success-100 text-success-700'
                }`}
              >
                {(detection.confidence * 100).toFixed(1)}%
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function TopDGADomains({ domains }: { domains: Stats['top_dga_domains'] }) {
  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Top DGA Domains</h3>
      {domains.length === 0 ? (
        <p className="text-gray-500 text-center py-4">No DGA domains detected</p>
      ) : (
        <div className="space-y-2">
          {domains.slice(0, 5).map((domain, index) => (
            <div key={domain.domain} className="flex items-center justify-between">
              <div className="flex items-center">
                <span className="w-6 h-6 flex items-center justify-center bg-danger-100 text-danger-700 rounded-full text-xs font-bold mr-3">
                  {index + 1}
                </span>
                <span className="font-mono text-sm">{domain.domain}</span>
              </div>
              <span className="text-sm text-gray-500">{domain.count} hits</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

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
    const interval = setInterval(fetchStats, 30000); // Refresh every 30s
    return () => clearInterval(interval);
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

  if (!stats) return null;

  // Prepare chart data
  const pieData = [
    { name: 'DGA', value: stats.dga_detected },
    { name: 'Legitimate', value: stats.legit_detected }
  ];

  const hourlyData = Object.entries(stats.hourly_stats).map(([hour, data]) => ({
    hour: hour.split(' ')[1],
    dga: data.dga,
    legit: data.legit,
    total: data.total
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <span className="text-sm text-gray-500">Last updated: {new Date().toLocaleTimeString()}</span>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Scans"
          value={stats.total_scans.toLocaleString()}
          icon={Activity}
          color="text-primary-600"
        />
        <StatCard
          title="DGA Detected"
          value={stats.dga_detected.toLocaleString()}
          icon={AlertTriangle}
          color="text-danger-600"
          subtitle={`${(stats.detection_rate * 100).toFixed(1)}% of total`}
        />
        <StatCard
          title="Legitimate"
          value={stats.legit_detected.toLocaleString()}
          icon={CheckCircle}
          color="text-success-600"
        />
        <StatCard
          title="Detection Rate"
          value={`${(stats.detection_rate * 100).toFixed(1)}%`}
          icon={TrendingUp}
          color="text-primary-600"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Distribution Pie Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Detection Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
              >
                <Cell fill={COLORS.dga} />
                <Cell fill={COLORS.legit} />
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly Activity Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Hourly Activity (24h)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip />
              <Bar dataKey="legit" stackId="a" fill={COLORS.legit} name="Legitimate" />
              <Bar dataKey="dga" stackId="a" fill={COLORS.dga} name="DGA" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RecentDetections detections={stats.recent_detections} />
        <TopDGADomains domains={stats.top_dga_domains} />
      </div>
    </div>
  );
}
