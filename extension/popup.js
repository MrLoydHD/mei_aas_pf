// Popup script for DGA Detector extension

document.addEventListener('DOMContentLoaded', async () => {
  // Elements
  const domainDisplay = document.getElementById('domain-display');
  const statusChecking = document.getElementById('status-checking');
  const statusSafe = document.getElementById('status-safe');
  const statusDanger = document.getElementById('status-danger');
  const statusNA = document.getElementById('status-na');
  const safeConfidence = document.getElementById('safe-confidence');
  const dangerConfidence = document.getElementById('danger-confidence');
  const statTotal = document.getElementById('stat-total');
  const statDGA = document.getElementById('stat-dga');
  const statSafe = document.getElementById('stat-safe');
  const manualDomain = document.getElementById('manual-domain');
  const checkBtn = document.getElementById('check-btn');
  const manualResult = document.getElementById('manual-result');

  // Auth elements
  const signInBtn = document.getElementById('sign-in-btn');
  const signOutBtn = document.getElementById('sign-out-btn');
  const userInfo = document.getElementById('user-info');
  const userAvatar = document.getElementById('user-avatar');
  const syncStatus = document.getElementById('sync-status');

  // Update auth UI
  function updateAuthUI(isSignedIn, user) {
    if (isSignedIn && user) {
      signInBtn.classList.add('hidden');
      userInfo.classList.remove('hidden');
      syncStatus.classList.remove('hidden');
      if (user.picture) {
        userAvatar.src = user.picture;
      } else {
        userAvatar.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%23fff"><circle cx="12" cy="8" r="4"/><path d="M12 14c-6 0-9 3-9 6v2h18v-2c0-3-3-6-9-6z"/></svg>';
      }
    } else {
      signInBtn.classList.remove('hidden');
      userInfo.classList.add('hidden');
      syncStatus.classList.add('hidden');
    }
  }

  // Check auth status on load
  chrome.runtime.sendMessage({ type: 'GET_AUTH_STATUS' }, (response) => {
    if (response) {
      updateAuthUI(response.isSignedIn, response.user);
    }
  });

  // Sign in handler
  signInBtn.addEventListener('click', async () => {
    signInBtn.disabled = true;
    signInBtn.textContent = '...';

    chrome.runtime.sendMessage({ type: 'SIGN_IN' }, (response) => {
      signInBtn.disabled = false;
      signInBtn.textContent = 'Sign In';

      if (response && response.success) {
        updateAuthUI(true, response.user);
        // Refresh stats after sign in
        chrome.runtime.sendMessage({ type: 'GET_STATS' }, (stats) => {
          if (stats) {
            updateStats(stats);
          }
        });
      } else {
        console.error('Sign in failed:', response?.error);
        // Show error to user
        alert('Sign in failed: ' + (response?.error || 'Unknown error'));
      }
    });
  });

  // Sign out handler
  signOutBtn.addEventListener('click', async () => {
    chrome.runtime.sendMessage({ type: 'SIGN_OUT' }, (response) => {
      if (response && response.success) {
        updateAuthUI(false, null);
      }
    });
  });

  // Hide all status indicators
  function hideAllStatus() {
    statusChecking.classList.add('hidden');
    statusSafe.classList.add('hidden');
    statusDanger.classList.add('hidden');
    statusNA.classList.add('hidden');
  }

  // Show specific status
  function showStatus(type, confidence = null) {
    hideAllStatus();

    switch (type) {
      case 'checking':
        statusChecking.classList.remove('hidden');
        break;
      case 'safe':
        statusSafe.classList.remove('hidden');
        if (confidence !== null) {
          safeConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
        }
        break;
      case 'danger':
        statusDanger.classList.remove('hidden');
        if (confidence !== null) {
          dangerConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
        }
        break;
      case 'na':
        statusNA.classList.remove('hidden');
        break;
    }
  }

  // Update statistics display
  function updateStats(stats) {
    statTotal.textContent = stats.totalChecked || 0;
    statDGA.textContent = stats.dgaDetected || 0;
    statSafe.textContent = stats.legitDetected || 0;
  }

  // Load statistics
  chrome.runtime.sendMessage({ type: 'GET_STATS' }, (stats) => {
    if (stats) {
      updateStats(stats);
    }
  });

  // Get current page status
  chrome.runtime.sendMessage({ type: 'GET_CURRENT_STATUS' }, (response) => {
    if (response && response.domain) {
      domainDisplay.textContent = response.domain;

      if (response.result) {
        if (response.result.is_dga) {
          showStatus('danger', response.result.confidence);
        } else {
          showStatus('safe', response.result.confidence);
        }
      } else {
        showStatus('na');
      }
    } else {
      domainDisplay.textContent = 'N/A';
      showStatus('na');
    }

    // Refresh stats after check
    chrome.runtime.sendMessage({ type: 'GET_STATS' }, (stats) => {
      if (stats) {
        updateStats(stats);
      }
    });
  });

  // Manual check handler
  async function performManualCheck() {
    const domain = manualDomain.value.trim();
    if (!domain) return;

    checkBtn.disabled = true;
    checkBtn.textContent = '...';
    manualResult.classList.remove('hidden');
    manualResult.innerHTML = '<div class="status checking"><div class="spinner"></div><div class="status-text"><strong>Checking...</strong></div></div>';

    chrome.runtime.sendMessage({ type: 'CHECK_DOMAIN', domain }, (result) => {
      checkBtn.disabled = false;
      checkBtn.textContent = 'Check';

      if (result) {
        const statusClass = result.is_dga ? 'danger' : 'safe';
        const statusText = result.is_dga ? 'DGA Detected!' : 'Safe Domain';
        const icon = result.is_dga
          ? '<svg class="status-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
          : '<svg class="status-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';

        manualResult.innerHTML = `
          <div class="status ${statusClass}">
            ${icon}
            <div class="status-text">
              <strong>${statusText}</strong>
              <span>Confidence: ${(result.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        `;

        // Update stats
        chrome.runtime.sendMessage({ type: 'GET_STATS' }, (stats) => {
          if (stats) {
            updateStats(stats);
          }
        });
      } else {
        manualResult.innerHTML = `
          <div class="status warning">
            <svg class="status-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <div class="status-text">
              <strong>Check Failed</strong>
              <span>Is the API running?</span>
            </div>
          </div>
        `;
      }
    });
  }

  checkBtn.addEventListener('click', performManualCheck);
  manualDomain.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      performManualCheck();
    }
  });
});
