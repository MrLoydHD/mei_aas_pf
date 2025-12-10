// Background service worker for DGA Detection extension

const API_URL = 'http://localhost:8000';
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes cache

// Domain cache to avoid repeated API calls
const domainCache = new Map();

// Statistics
let stats = {
  totalChecked: 0,
  dgaDetected: 0,
  legitDetected: 0,
  lastChecked: null
};

// Load stats from storage on startup
chrome.storage.local.get(['dgaStats'], (result) => {
  if (result.dgaStats) {
    stats = result.dgaStats;
  }
});

// Extract domain from URL
function extractDomain(url) {
  try {
    const urlObj = new URL(url);
    let domain = urlObj.hostname;

    // Remove www prefix
    if (domain.startsWith('www.')) {
      domain = domain.substring(4);
    }

    return domain;
  } catch {
    return null;
  }
}

// Check if domain should be skipped
function shouldSkipDomain(domain) {
  const skipPatterns = [
    'localhost',
    '127.0.0.1',
    'chrome.',
    'google.com',
    'googleapis.com',
    'gstatic.com',
    'chrome-extension',
    'extension',
    // Common CDNs and trusted domains
    'cloudflare.com',
    'amazonaws.com',
    'akamai.net',
    'fastly.net'
  ];

  return skipPatterns.some(pattern => domain.includes(pattern));
}

// Check domain with API
async function checkDomain(domain) {
  // Check cache first
  const cached = domainCache.get(domain);
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    return cached.result;
  }

  try {
    const response = await fetch(`${API_URL}/extension/check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ domain })
    });

    if (!response.ok) {
      throw new Error('API request failed');
    }

    const result = await response.json();

    // Cache the result
    domainCache.set(domain, {
      result,
      timestamp: Date.now()
    });

    // Update stats
    stats.totalChecked++;
    if (result.is_dga) {
      stats.dgaDetected++;
    } else {
      stats.legitDetected++;
    }
    stats.lastChecked = new Date().toISOString();

    // Save stats
    chrome.storage.local.set({ dgaStats: stats });

    return result;
  } catch (error) {
    console.error('DGA check failed:', error);
    return null;
  }
}

// Update badge based on detection
function updateBadge(tabId, result) {
  if (!result) {
    chrome.action.setBadgeText({ tabId, text: '' });
    return;
  }

  if (result.is_dga) {
    chrome.action.setBadgeText({ tabId, text: '!' });
    chrome.action.setBadgeBackgroundColor({ tabId, color: '#ef4444' });
  } else if (result.risk_level === 'low') {
    chrome.action.setBadgeText({ tabId, text: '✓' });
    chrome.action.setBadgeBackgroundColor({ tabId, color: '#22c55e' });
  } else {
    chrome.action.setBadgeText({ tabId, text: '?' });
    chrome.action.setBadgeBackgroundColor({ tabId, color: '#f59e0b' });
  }
}

// Handle navigation events
chrome.webNavigation.onCompleted.addListener(async (details) => {
  // Only check main frame
  if (details.frameId !== 0) return;

  const domain = extractDomain(details.url);
  if (!domain || shouldSkipDomain(domain)) return;

  const result = await checkDomain(domain);
  updateBadge(details.tabId, result);

  // Send result to content script if DGA detected
  if (result && result.is_dga) {
    try {
      await chrome.tabs.sendMessage(details.tabId, {
        type: 'DGA_DETECTED',
        domain,
        result
      });
    } catch {
      // Content script might not be ready
    }
  }
});

// Handle messages from popup and content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'CHECK_DOMAIN') {
    checkDomain(message.domain).then(result => {
      sendResponse(result);
    });
    return true; // Async response
  }

  if (message.type === 'GET_STATS') {
    sendResponse(stats);
    return true;
  }

  if (message.type === 'GET_CURRENT_STATUS') {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      if (tabs[0] && tabs[0].url) {
        const domain = extractDomain(tabs[0].url);
        if (domain && !shouldSkipDomain(domain)) {
          const result = await checkDomain(domain);
          sendResponse({ domain, result });
        } else {
          sendResponse({ domain: null, result: null });
        }
      } else {
        sendResponse({ domain: null, result: null });
      }
    });
    return true;
  }

  if (message.type === 'CLEAR_STATS') {
    stats = {
      totalChecked: 0,
      dgaDetected: 0,
      legitDetected: 0,
      lastChecked: null
    };
    chrome.storage.local.set({ dgaStats: stats });
    sendResponse({ success: true });
    return true;
  }
});

// Clean up old cache entries periodically
setInterval(() => {
  const now = Date.now();
  for (const [domain, entry] of domainCache.entries()) {
    if (now - entry.timestamp > CACHE_DURATION) {
      domainCache.delete(domain);
    }
  }
}, 60000); // Every minute

console.log('DGA Detector background service started');
