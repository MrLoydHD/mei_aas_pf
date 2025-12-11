import { Card, CardContent } from '@/components/ui/card';

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ElementType;
  variant: 'primary' | 'danger' | 'success';
  subtitle?: string;
}

export function StatCard({ title, value, icon: Icon, variant, subtitle }: StatCardProps) {
  const colorClasses = {
    primary: 'text-primary',
    danger: 'text-destructive',
    success: 'text-success-500'
  };

  const bgClasses = {
    primary: 'bg-primary/10',
    danger: 'bg-destructive/10',
    success: 'bg-success-100'
  };

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <p className={`text-3xl font-bold ${colorClasses[variant]}`}>{value}</p>
            {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
          </div>
          <div className={`p-3 rounded-full ${bgClasses[variant]}`}>
            <Icon className={`h-6 w-6 ${colorClasses[variant]}`} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
