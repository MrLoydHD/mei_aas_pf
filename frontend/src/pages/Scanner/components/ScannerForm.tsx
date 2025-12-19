import { Search, Loader2, Info, Shield } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BINARY_MODEL_OPTIONS, BinaryModelType } from '@/types';

interface ScannerFormProps {
  mode: 'single' | 'batch';
  onModeChange: (mode: 'single' | 'batch') => void;
  domain: string;
  onDomainChange: (value: string) => void;
  batchInput: string;
  onBatchInputChange: (value: string) => void;
  modelType: string;
  onModelTypeChange: (value: string) => void;
  detailed: boolean;
  onDetailedChange: (value: boolean) => void;
  includeFamily: boolean;
  onIncludeFamilyChange: (value: boolean) => void;
  loading: boolean;
  onSingleScan: () => void;
  onBatchScan: () => void;
}

export function ScannerForm({
  mode,
  onModeChange,
  domain,
  onDomainChange,
  batchInput,
  onBatchInputChange,
  modelType,
  onModelTypeChange,
  detailed,
  onDetailedChange,
  includeFamily,
  onIncludeFamilyChange,
  loading,
  onSingleScan,
  onBatchScan
}: ScannerFormProps) {
  return (
    <Card>
      <CardContent className="pt-6">
        <Tabs value={mode} onValueChange={(v) => onModeChange(v as 'single' | 'batch')}>
          <TabsList className="mb-4">
            <TabsTrigger value="single">Single Domain</TabsTrigger>
            <TabsTrigger value="batch">Batch Scan</TabsTrigger>
          </TabsList>

          <div className="flex flex-wrap items-center gap-4 mb-4">
            <label className="flex items-center">
              <span className="text-sm text-muted-foreground mr-2">Model:</span>
              <select
                value={modelType}
                onChange={(e) => onModelTypeChange(e.target.value as BinaryModelType)}
                className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {BINARY_MODEL_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value} title={option.description}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            {mode === 'single' && (
              <>
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeFamily}
                    onChange={(e) => {
                      onIncludeFamilyChange(e.target.checked);
                      if (e.target.checked) onDetailedChange(false);
                    }}
                    className="mr-2 h-4 w-4 rounded border-input"
                  />
                  <Shield className="h-4 w-4 mr-1 text-destructive" />
                  <span className="text-sm text-muted-foreground">Family classification</span>
                </label>

                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={detailed}
                    onChange={(e) => {
                      onDetailedChange(e.target.checked);
                      if (e.target.checked) onIncludeFamilyChange(false);
                    }}
                    className="mr-2 h-4 w-4 rounded border-input"
                  />
                  <span className="text-sm text-muted-foreground">Show features (RF only)</span>
                </label>
              </>
            )}
          </div>

          <TabsContent value="single">
            <div className="flex space-x-4">
              <Input
                type="text"
                value={domain}
                onChange={(e) => onDomainChange(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && onSingleScan()}
                placeholder="Enter domain (e.g., google.com or suspicious123abc.net)"
                className="flex-1"
              />
              <Button
                onClick={onSingleScan}
                disabled={loading || !domain.trim()}
              >
                {loading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <>
                    <Search className="h-5 w-5 mr-2" />
                    Scan
                  </>
                )}
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="batch">
            <div className="space-y-4">
              <textarea
                value={batchInput}
                onChange={(e) => onBatchInputChange(e.target.value)}
                placeholder="Enter domains (one per line)&#10;google.com&#10;suspicious123abc.net&#10;facebook.com"
                className="flex min-h-40 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring font-mono"
              />
              <Button
                onClick={onBatchScan}
                disabled={loading || !batchInput.trim()}
              >
                {loading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <>
                    <Search className="h-5 w-5 mr-2" />
                    Scan All
                  </>
                )}
              </Button>
            </div>
          </TabsContent>
        </Tabs>

        <div className="mt-4 flex items-start space-x-2 text-sm text-muted-foreground">
          <Info className="h-4 w-4 mt-0.5 flex-shrink-0" />
          <p>
            Enter a domain name or URL. The system will extract the domain and analyze it for DGA patterns.
            With <strong>Family classification</strong> enabled, detected DGAs will be identified by malware family
            (e.g., Conficker, Emotet) with threat intelligence.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
