/**
 * k6 Smoke Test Script for DGA Detection API
 *
 * Quick validation that the API is working correctly.
 *
 * Usage:
 *   k6 run tests/load/k6-smoke-test.js
 */

import http from 'k6/http';
import { check, group } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  vus: 1,
  iterations: 10,
  thresholds: {
    http_req_duration: ['p(95)<1000'],
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.95'],
  },
};

export default function () {
  const headers = { 'Content-Type': 'application/json' };

  // 1. Health Check
  group('Health Check', function () {
    const res = http.get(`${BASE_URL}/health`);
    check(res, {
      'health status 200': (r) => r.status === 200,
      'health is healthy': (r) => JSON.parse(r.body).status === 'healthy',
    });
  });

  // 2. Single Prediction - Legit Domain
  group('Predict Legit Domain', function () {
    const res = http.post(
      `${BASE_URL}/predict?log=false`,
      JSON.stringify({ domain: 'google.com' }),
      { headers }
    );
    check(res, {
      'predict status 200': (r) => r.status === 200,
      'google.com not DGA': (r) => JSON.parse(r.body).is_dga === false,
    });
  });

  // 3. Single Prediction - DGA Domain
  group('Predict DGA Domain', function () {
    const res = http.post(
      `${BASE_URL}/predict?log=false`,
      JSON.stringify({ domain: 'asdfjkl123xyz.net' }),
      { headers }
    );
    check(res, {
      'predict status 200': (r) => r.status === 200,
      'random domain is DGA': (r) => JSON.parse(r.body).is_dga === true,
    });
  });

  // 4. Batch Prediction
  group('Batch Prediction', function () {
    const domains = ['google.com', 'xkl93jfsd.net', 'facebook.com'];
    const res = http.post(
      `${BASE_URL}/predict/batch?log=false`,
      JSON.stringify({ domains }),
      { headers }
    );
    check(res, {
      'batch status 200': (r) => r.status === 200,
      'batch has 3 results': (r) => JSON.parse(r.body).total === 3,
    });
  });

  // 5. Family Prediction
  group('Family Prediction', function () {
    const res = http.post(
      `${BASE_URL}/predict/family?log=false`,
      JSON.stringify({ domain: 'asdfjkl123xyz.net' }),
      { headers }
    );
    check(res, {
      'family status 200': (r) => r.status === 200,
      'has is_dga field': (r) => JSON.parse(r.body).is_dga !== undefined,
    });
  });

  // 6. Extension Check
  group('Extension Check', function () {
    const res = http.post(
      `${BASE_URL}/extension/check`,
      JSON.stringify({ domain: 'github.com' }),
      { headers }
    );
    check(res, {
      'extension status 200': (r) => r.status === 200,
      'has risk_level': (r) => JSON.parse(r.body).risk_level !== undefined,
    });
  });

  // 7. Statistics
  group('Statistics', function () {
    const res = http.get(`${BASE_URL}/stats`);
    check(res, {
      'stats status 200': (r) => r.status === 200,
    });
  });

  // 8. Models Info
  group('Models Info', function () {
    const res = http.get(`${BASE_URL}/models`);
    check(res, {
      'models status 200': (r) => r.status === 200,
      'has models list': (r) => JSON.parse(r.body).models !== undefined,
    });
  });

  // 9. Families Info
  group('Families Info', function () {
    const res = http.get(`${BASE_URL}/families`);
    check(res, {
      'families status 200': (r) => r.status === 200,
      'has families': (r) => JSON.parse(r.body).families !== undefined,
    });
  });

  // 10. Metrics Endpoint
  group('Metrics', function () {
    const res = http.get(`${BASE_URL}/metrics`);
    check(res, {
      'metrics status 200': (r) => r.status === 200,
      'has prometheus metrics': (r) => r.body.includes('dga_predictions_total'),
    });
  });
}
