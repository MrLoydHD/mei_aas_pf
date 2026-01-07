/**
 * k6 Load Test Script for DGA Detection API
 *
 * Installation:
 *   - macOS: brew install k6
 *   - Linux: sudo snap install k6
 *   - Windows: choco install k6
 *   - Docker: docker run --rm -i grafana/k6 run - < k6-load-test.js
 *
 * Usage:
 *   Basic run:     k6 run k6-load-test.js
 *   With options:  k6 run --vus 50 --duration 60s k6-load-test.js
 *   Smoke test:    k6 run --env TEST_TYPE=smoke k6-load-test.js
 *   Load test:     k6 run --env TEST_TYPE=load k6-load-test.js
 *   Stress test:   k6 run --env TEST_TYPE=stress k6-load-test.js
 *
 * Environment variables:
 *   BASE_URL: API base URL (default: http://localhost:8000)
 *   TEST_TYPE: smoke, load, or stress (default: load)
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const dgaDetectionRate = new Rate('dga_detection_rate');
const predictionLatency = new Trend('prediction_latency');
const batchLatency = new Trend('batch_prediction_latency');
const familyLatency = new Trend('family_prediction_latency');
const errorCount = new Counter('errors');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const TEST_TYPE = __ENV.TEST_TYPE || 'load';

// Test scenarios
const testConfigs = {
  smoke: {
    vus: 1,
    duration: '30s',
    thresholds: {
      http_req_duration: ['p(95)<500'],
      http_req_failed: ['rate<0.01'],
    },
  },
  load: {
    stages: [
      { duration: '1m', target: 10 },  // Ramp up
      { duration: '3m', target: 10 },  // Steady state
      { duration: '1m', target: 20 },  // Peak
      { duration: '2m', target: 20 },  // Sustained peak
      { duration: '1m', target: 0 },   // Ramp down
    ],
    thresholds: {
      http_req_duration: ['p(95)<1000', 'p(99)<2000'],
      http_req_failed: ['rate<0.05'],
      prediction_latency: ['p(95)<500'],
    },
  },
  stress: {
    stages: [
      { duration: '2m', target: 20 },
      { duration: '3m', target: 50 },
      { duration: '2m', target: 100 },
      { duration: '3m', target: 100 },
      { duration: '2m', target: 0 },
    ],
    thresholds: {
      http_req_duration: ['p(95)<2000'],
      http_req_failed: ['rate<0.10'],
    },
  },
};

// Export options based on test type
export const options = {
  ...testConfigs[TEST_TYPE],
  thresholds: {
    ...testConfigs[TEST_TYPE].thresholds,
    dga_detection_rate: ['rate>0'],  // Just track, no threshold
  },
};

// Sample domains for testing
const legitDomains = [
  'google.com',
  'facebook.com',
  'youtube.com',
  'amazon.com',
  'twitter.com',
  'instagram.com',
  'linkedin.com',
  'netflix.com',
  'reddit.com',
  'github.com',
  'stackoverflow.com',
  'microsoft.com',
  'apple.com',
  'wikipedia.org',
  'cloudflare.com',
];

const dgaDomains = [
  'asdkjfhwer.com',
  'xkl93jfsd.net',
  'qwerty123abc.org',
  'asdf8923jkl.com',
  'zxcvbn456def.net',
  'mnbvc7890ghi.org',
  'poiuyt135jkl.com',
  'lkjhgf246mno.net',
  'wertyu357pqr.org',
  'cvbnm468stu.com',
  'fghjk579vwx.net',
  'tyuio680yza.org',
  'dfghj791bcd.com',
  'sdfgh802efg.net',
  'xcvbn913hij.org',
];

// Helper function to get random domain
function getRandomDomain(type = 'mixed') {
  let domains;
  if (type === 'legit') {
    domains = legitDomains;
  } else if (type === 'dga') {
    domains = dgaDomains;
  } else {
    domains = Math.random() > 0.5 ? legitDomains : dgaDomains;
  }
  return domains[Math.floor(Math.random() * domains.length)];
}

// Helper function to get random batch
function getRandomBatch(size = 10) {
  const domains = [];
  for (let i = 0; i < size; i++) {
    domains.push(getRandomDomain());
  }
  return domains;
}

// Setup function - runs once before the test
export function setup() {
  console.log(`Running ${TEST_TYPE} test against ${BASE_URL}`);

  // Health check
  const healthRes = http.get(`${BASE_URL}/health`);
  if (healthRes.status !== 200) {
    throw new Error(`API not healthy: ${healthRes.status}`);
  }

  const healthData = JSON.parse(healthRes.body);
  console.log(`API Health: ${healthData.status}`);
  console.log(`Models loaded: ${JSON.stringify(healthData.models_loaded)}`);

  return { startTime: Date.now() };
}

// Main test function
export default function () {
  const headers = {
    'Content-Type': 'application/json',
  };

  // Group: Health Check
  group('Health Check', function () {
    const res = http.get(`${BASE_URL}/health`);
    check(res, {
      'health status is 200': (r) => r.status === 200,
      'health response is healthy': (r) => JSON.parse(r.body).status === 'healthy',
    });
  });

  // Group: Single Domain Prediction
  group('Single Domain Prediction', function () {
    const domain = getRandomDomain();
    const payload = JSON.stringify({ domain: domain });

    const startTime = Date.now();
    const res = http.post(`${BASE_URL}/predict?log=false`, payload, { headers });
    const latency = Date.now() - startTime;

    predictionLatency.add(latency);

    const success = check(res, {
      'predict status is 200': (r) => r.status === 200,
      'predict has is_dga field': (r) => JSON.parse(r.body).is_dga !== undefined,
      'predict has confidence field': (r) => JSON.parse(r.body).confidence !== undefined,
      'predict latency < 500ms': () => latency < 500,
    });

    if (!success) {
      errorCount.add(1);
    }

    if (res.status === 200) {
      const data = JSON.parse(res.body);
      dgaDetectionRate.add(data.is_dga);
    }
  });

  sleep(0.1);  // Small pause between requests

  // Group: Batch Prediction (less frequent)
  if (Math.random() < 0.3) {  // 30% of iterations
    group('Batch Prediction', function () {
      const domains = getRandomBatch(5);
      const payload = JSON.stringify({ domains: domains });

      const startTime = Date.now();
      const res = http.post(`${BASE_URL}/predict/batch?log=false`, payload, { headers });
      const latency = Date.now() - startTime;

      batchLatency.add(latency);

      const success = check(res, {
        'batch status is 200': (r) => r.status === 200,
        'batch has predictions': (r) => JSON.parse(r.body).predictions !== undefined,
        'batch count matches': (r) => JSON.parse(r.body).total === domains.length,
        'batch latency < 2000ms': () => latency < 2000,
      });

      if (!success) {
        errorCount.add(1);
      }
    });
  }

  sleep(0.1);

  // Group: Family Prediction (less frequent)
  if (Math.random() < 0.2) {  // 20% of iterations
    group('Family Prediction', function () {
      const domain = getRandomDomain('dga');  // Use DGA domain for family classification
      const payload = JSON.stringify({ domain: domain });

      const startTime = Date.now();
      const res = http.post(`${BASE_URL}/predict/family?log=false`, payload, { headers });
      const latency = Date.now() - startTime;

      familyLatency.add(latency);

      const success = check(res, {
        'family status is 200': (r) => r.status === 200,
        'family has is_dga field': (r) => JSON.parse(r.body).is_dga !== undefined,
        'family latency < 1000ms': () => latency < 1000,
      });

      if (!success) {
        errorCount.add(1);
      }
    });
  }

  sleep(0.1);

  // Group: Extension Check (simulates browser extension)
  if (Math.random() < 0.4) {  // 40% of iterations
    group('Extension Check', function () {
      const domain = getRandomDomain();
      const payload = JSON.stringify({ domain: domain });

      const startTime = Date.now();
      const res = http.post(`${BASE_URL}/extension/check`, payload, { headers });
      const latency = Date.now() - startTime;

      check(res, {
        'extension status is 200': (r) => r.status === 200,
        'extension has is_dga': (r) => JSON.parse(r.body).is_dga !== undefined,
        'extension has risk_level': (r) => JSON.parse(r.body).risk_level !== undefined,
        'extension latency < 200ms': () => latency < 200,  // Extension needs to be fast
      });
    });
  }

  sleep(0.2);  // Think time between requests

  // Group: Stats endpoint (occasional)
  if (Math.random() < 0.1) {  // 10% of iterations
    group('Statistics', function () {
      const res = http.get(`${BASE_URL}/stats`);
      check(res, {
        'stats status is 200': (r) => r.status === 200,
        'stats has total_scans': (r) => JSON.parse(r.body).total_scans !== undefined,
      });
    });
  }

  // Group: Models info (occasional)
  if (Math.random() < 0.05) {  // 5% of iterations
    group('Models Info', function () {
      const res = http.get(`${BASE_URL}/models`);
      check(res, {
        'models status is 200': (r) => r.status === 200,
        'models has list': (r) => JSON.parse(r.body).models !== undefined,
      });
    });
  }
}

// Teardown function - runs once after the test
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Test completed in ${duration.toFixed(2)} seconds`);
}

// Handle summary
export function handleSummary(data) {
  const summary = {
    testType: TEST_TYPE,
    baseUrl: BASE_URL,
    timestamp: new Date().toISOString(),
    metrics: {
      http_reqs: data.metrics.http_reqs?.values,
      http_req_duration: data.metrics.http_req_duration?.values,
      http_req_failed: data.metrics.http_req_failed?.values,
      prediction_latency: data.metrics.prediction_latency?.values,
      batch_prediction_latency: data.metrics.batch_prediction_latency?.values,
      dga_detection_rate: data.metrics.dga_detection_rate?.values,
      errors: data.metrics.errors?.values,
    },
  };

  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'tests/load/results.json': JSON.stringify(summary, null, 2),
  };
}

// Text summary helper
function textSummary(data, options) {
  const lines = [];
  lines.push('\n========== DGA Detection API Load Test Summary ==========\n');
  lines.push(`Test Type: ${TEST_TYPE}`);
  lines.push(`Base URL: ${BASE_URL}`);
  lines.push('');

  // Key metrics
  if (data.metrics.http_reqs) {
    lines.push(`Total Requests: ${data.metrics.http_reqs.values.count}`);
    lines.push(`Requests/sec: ${data.metrics.http_reqs.values.rate.toFixed(2)}`);
  }

  if (data.metrics.http_req_duration) {
    const dur = data.metrics.http_req_duration.values;
    lines.push(`Response Time (avg): ${dur.avg.toFixed(2)}ms`);
    lines.push(`Response Time (p95): ${dur['p(95)'].toFixed(2)}ms`);
    lines.push(`Response Time (p99): ${dur['p(99)'].toFixed(2)}ms`);
  }

  if (data.metrics.http_req_failed) {
    const failed = data.metrics.http_req_failed.values;
    lines.push(`Error Rate: ${(failed.rate * 100).toFixed(2)}%`);
  }

  if (data.metrics.prediction_latency) {
    const lat = data.metrics.prediction_latency.values;
    lines.push(`Prediction Latency (avg): ${lat.avg.toFixed(2)}ms`);
    lines.push(`Prediction Latency (p95): ${lat['p(95)'].toFixed(2)}ms`);
  }

  if (data.metrics.dga_detection_rate) {
    const rate = data.metrics.dga_detection_rate.values;
    lines.push(`DGA Detection Rate: ${(rate.rate * 100).toFixed(2)}%`);
  }

  lines.push('\n==========================================================\n');

  return lines.join('\n');
}
