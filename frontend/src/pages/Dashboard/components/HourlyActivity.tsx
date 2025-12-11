import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { Stats } from '@/types';

interface HourlyActivityProps {
  hourlyStats: Stats['hourly_stats'];
}

const COLORS = {
  dga: '#ef4444',
  legit: '#22c55e'
};

export function HourlyActivity({ hourlyStats }: HourlyActivityProps) {
  const hourlyData = Object.entries(hourlyStats).map(([hour, data]) => ({
    hour: hour.split(' ')[1],
    dga: data.dga,
    legit: data.legit,
    total: data.total
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Hourly Activity (24h)</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={hourlyData}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis dataKey="hour" fontSize={12} className="fill-muted-foreground" />
            <YAxis fontSize={12} className="fill-muted-foreground" />
            <Tooltip />
            <Bar dataKey="legit" stackId="a" fill={COLORS.legit} name="Legitimate" />
            <Bar dataKey="dga" stackId="a" fill={COLORS.dga} name="DGA" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
