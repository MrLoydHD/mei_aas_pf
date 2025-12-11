import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface DetectionDistributionProps {
  dgaCount: number;
  legitCount: number;
}

const COLORS = {
  dga: '#ef4444',
  legit: '#22c55e'
};

export function DetectionDistribution({ dgaCount, legitCount }: DetectionDistributionProps) {
  const pieData = [
    { name: 'DGA', value: dgaCount },
    { name: 'Legitimate', value: legitCount }
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Detection Distribution</CardTitle>
      </CardHeader>
      <CardContent>
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
      </CardContent>
    </Card>
  );
}
