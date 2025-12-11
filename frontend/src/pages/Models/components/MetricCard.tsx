import { Progress } from '@/components/ui/progress';

interface MetricCardProps {
  label: string;
  value: number;
  color: string;
}

// Map text colors to background colors (explicit for Tailwind to include them)
const colorMap: Record<string, string> = {
  'text-primary': 'bg-primary',
  'text-blue-600': 'bg-blue-600',
  'text-purple-600': 'bg-purple-600',
  'text-indigo-600': 'bg-indigo-600',
  'text-green-600': 'bg-green-600',
};

export function MetricCard({ label, value, color }: MetricCardProps) {
  const percentage = value * 100;
  const indicatorColor = colorMap[color] || 'bg-primary';

  return (
    <div className="bg-muted rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-muted-foreground">{label}</span>
        <span className={`text-lg font-bold ${color}`}>{percentage.toFixed(2)}%</span>
      </div>
      <Progress value={percentage} className="h-2" indicatorClassName={indicatorColor} />
    </div>
  );
}
