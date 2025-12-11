import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { Stats } from '@/types';

interface TopDGADomainsProps {
  domains: Stats['top_dga_domains'];
}

export function TopDGADomains({ domains }: TopDGADomainsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Top DGA Domains</CardTitle>
      </CardHeader>
      <CardContent>
        {domains.length === 0 ? (
          <p className="text-muted-foreground text-center py-4">No DGA domains detected</p>
        ) : (
          <div className="space-y-2">
            {domains.slice(0, 5).map((domain, index) => (
              <div key={domain.domain} className="flex items-center justify-between">
                <div className="flex items-center">
                  <span className="w-6 h-6 flex items-center justify-center bg-destructive/10 text-destructive rounded-full text-xs font-bold mr-3">
                    {index + 1}
                  </span>
                  <span className="font-mono text-sm">{domain.domain}</span>
                </div>
                <span className="text-sm text-muted-foreground">{domain.count} hits</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
