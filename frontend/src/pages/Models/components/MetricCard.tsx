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
  const safeValue = value ?? 0;
  const percentage = safeValue * 100;
  const indicatorColor = colorMap[color] || 'bg-primary';
  const isValid = value !== undefined && value !== null && !isNaN(value);

  return (
    <div className="bg-muted rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-muted-foreground">{label}</span>
        <span className={`text-lg font-bold ${color}`}>
          {isValid ? `${percentage.toFixed(2)}%` : 'N/A'}
        </span>
      </div>
      <Progress value={isValid ? percentage : 0} className="h-2" indicatorClassName={indicatorColor} />
    </div>
  );
}
