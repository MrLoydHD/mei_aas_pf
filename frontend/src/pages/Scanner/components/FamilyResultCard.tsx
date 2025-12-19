import { AlertTriangle, CheckCircle, Shield, Bug, Calendar, Info } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { FamilyPredictionResult } from '@/types';

interface FamilyResultCardProps {
  result: FamilyPredictionResult;
}

const threatLevelColors: Record<string, string> = {
  critical: 'bg-red-600 text-white',
  high: 'bg-orange-500 text-white',
  medium: 'bg-yellow-500 text-black',
  low: 'bg-green-500 text-white',
  unknown: 'bg-gray-500 text-white',
};

const malwareTypeIcons: Record<string, string> = {
  ransomware: 'Ransomware',
  banking_trojan: 'Banking Trojan',
  botnet: 'Botnet',
  worm: 'Worm',
  trojan: 'Trojan',
  backdoor: 'Backdoor',
  bootkit: 'Bootkit',
  adware: 'Adware',
};

export function FamilyResultCard({ result }: FamilyResultCardProps) {
  const hasFamilyInfo = result.is_dga && result.family_info;

  return (
    <Card className={`border-2 ${result.is_dga ? 'border-destructive bg-destructive/5' : 'border-success-500 bg-success-50'}`}>
      <CardContent className="pt-6">
        {/* Header Section */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center">
            {result.is_dga ? (
              <AlertTriangle className="h-8 w-8 text-destructive mr-3" />
            ) : (
              <CheckCircle className="h-8 w-8 text-success-500 mr-3" />
            )}
            <div>
              <h3 className="text-xl font-bold text-foreground">{result.domain}</h3>
              <p className={`text-sm ${result.is_dga ? 'text-destructive' : 'text-success-600'}`}>
                {result.is_dga ? 'DGA Detected - Potentially Malicious' : 'Likely Legitimate'}
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold text-foreground">{(result.dga_confidence * 100).toFixed(1)}%</p>
            <p className="text-sm text-muted-foreground">DGA Confidence</p>
          </div>
        </div>

        {/* Basic Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div className="bg-card rounded-lg p-3 border">
            <p className="text-xs text-muted-foreground">DGA Model</p>
            <p className="text-lg font-semibold text-foreground capitalize">{result.model_used}</p>
          </div>
          <div className="bg-card rounded-lg p-3 border">
            <p className="text-xs text-muted-foreground">Family Model</p>
            <p className="text-lg font-semibold text-foreground">
              {result.family_model_used === 'ensemble' ? 'Ensemble' :
               result.family_model_used === 'family_classifier_lstm' ? 'LSTM' :
               result.family_model_used === 'family_classifier_rf' ? 'RF' :
               result.family_model_used ? result.family_model_used : 'N/A'}
            </p>
          </div>
          <div className="bg-card rounded-lg p-3 border">
            <p className="text-xs text-muted-foreground">Status</p>
            <p className={`text-lg font-semibold ${result.is_dga ? 'text-destructive' : 'text-success-600'}`}>
              {result.is_dga ? 'Malicious' : 'Safe'}
            </p>
          </div>
          <div className="bg-card rounded-lg p-3 border">
            <p className="text-xs text-muted-foreground">Analyzed At</p>
            <p className="text-sm font-semibold text-foreground">{new Date(result.timestamp).toLocaleTimeString()}</p>
          </div>
        </div>

        {/* Family Information Section */}
        {hasFamilyInfo && result.family_info && (
          <div className="mt-6 border-t pt-4">
            <div className="flex items-center mb-4">
              <Shield className="h-5 w-5 text-destructive mr-2" />
              <h4 className="text-lg font-semibold text-foreground">Threat Intelligence</h4>
            </div>

            {/* Family Header */}
            <div className="bg-destructive/10 rounded-lg p-4 mb-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <Bug className="h-6 w-6 text-destructive mr-2" />
                  <span className="text-xl font-bold text-foreground capitalize">
                    {result.family_info.family}
                  </span>
                </div>
                <Badge className={threatLevelColors[result.family_info.threat_level]}>
                  {result.family_info.threat_level.toUpperCase()} THREAT
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                {result.family_info.description}
              </p>
            </div>

            {/* Family Details Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div className="bg-card rounded-lg p-3 border">
                <p className="text-xs text-muted-foreground">Malware Type</p>
                <p className="text-sm font-semibold text-foreground">
                  {malwareTypeIcons[result.family_info.malware_type] || result.family_info.malware_type}
                </p>
              </div>
              <div className="bg-card rounded-lg p-3 border">
                <div className="flex items-center text-xs text-muted-foreground mb-1">
                  <Calendar className="h-3 w-3 mr-1" />
                  First Seen
                </div>
                <p className="text-sm font-semibold text-foreground">{result.family_info.first_seen}</p>
              </div>
              <div className="bg-card rounded-lg p-3 border">
                <p className="text-xs text-muted-foreground">Family Confidence</p>
                <p className="text-lg font-semibold text-foreground">
                  {(result.family_info.confidence * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-card rounded-lg p-3 border">
                <p className="text-xs text-muted-foreground">Threat Level</p>
                <p className={`text-sm font-semibold capitalize ${
                  result.family_info.threat_level === 'critical' ? 'text-red-600' :
                  result.family_info.threat_level === 'high' ? 'text-orange-500' :
                  result.family_info.threat_level === 'medium' ? 'text-yellow-600' :
                  'text-green-600'
                }`}>
                  {result.family_info.threat_level}
                </p>
              </div>
            </div>

            {/* Alternative Families */}
            {result.family_info.alternatives && result.family_info.alternatives.length > 0 && (
              <div className="mt-4">
                <div className="flex items-center mb-2">
                  <Info className="h-4 w-4 text-muted-foreground mr-1" />
                  <h5 className="text-sm font-semibold text-muted-foreground">Other Possible Families</h5>
                </div>
                <div className="flex flex-wrap gap-2">
                  {result.family_info.alternatives.map((alt, idx) => (
                    <Badge key={idx} variant="outline" className="text-xs">
                      {alt.family}: {(alt.confidence * 100).toFixed(1)}%
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* No Family Info Message */}
        {result.is_dga && !hasFamilyInfo && (
          <div className="mt-4 p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">
              Family classification not available. The family classifier model may not be loaded.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
