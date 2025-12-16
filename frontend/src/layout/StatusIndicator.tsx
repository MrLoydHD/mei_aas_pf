import { Activity } from 'lucide-react';

export function StatusIndicator() {
  return (
    <div className="fixed bottom-4 right-4 flex items-center bg-card/90 backdrop-blur-sm rounded-full shadow-lg px-4 py-2 border border-border z-10">
      <Activity className="h-4 w-4 text-success-500 mr-2 animate-pulse" />
      <span className="text-sm text-muted-foreground">API Connected</span>
    </div>
  );
}
