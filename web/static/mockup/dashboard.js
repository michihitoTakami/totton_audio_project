// ============================================================================
// Mock Headphone Search (no API, just show results)
// ============================================================================
const searchInput = document.getElementById('headphoneSearch');
const searchResults = document.getElementById('searchResults');
const currentHeadphone = document.getElementById('currentHeadphone');

searchInput.addEventListener('input', (e) => {
    const query = e.target.value.trim();
    if (query.length >= 2) {
        searchResults.style.display = 'block';
    } else {
        searchResults.style.display = 'none';
    }
});

// Select headphone
document.querySelectorAll('.search-result-item').forEach(item => {
    item.addEventListener('click', () => {
        const name = item.dataset.name;
        currentHeadphone.textContent = name;
        searchInput.value = '';
        searchResults.style.display = 'none';
        showToast('ヘッドホンを選択しました: ' + name);
    });
});

// Hide search on click outside
document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
        searchResults.style.display = 'none';
    }
});

// ============================================================================
// Crossfeed Toggle
// ============================================================================
const crossfeedToggle = document.getElementById('crossfeedToggle');
const headSizeSection = document.getElementById('headSizeSection');
const crossfeedStatus = document.getElementById('crossfeedStatus');
const lowLatencyToggle = document.getElementById('lowLatencyToggle');
const processingMode = document.getElementById('processingMode');

const headSizeButtons = document.querySelectorAll('.control-btn[data-size]');

crossfeedToggle.addEventListener('change', (e) => {
    if (e.target.checked) {
        headSizeSection.style.display = 'block';
        const activeSize = document.querySelector('.control-btn[data-size].active').dataset.size.toUpperCase();
        crossfeedStatus.textContent = 'ON (' + activeSize + ')';
        // Disable low latency when crossfeed is on
        lowLatencyToggle.checked = false;
        lowLatencyToggle.disabled = true;
        processingMode.textContent = 'Standard';
        showToast('クロスフィードを有効化しました');
    } else {
        headSizeSection.style.display = 'none';
        crossfeedStatus.textContent = 'OFF';
        lowLatencyToggle.disabled = false;
        showToast('クロスフィードを無効化しました');
    }
});

headSizeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        headSizeButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const size = btn.dataset.size.toUpperCase();
        crossfeedStatus.textContent = 'ON (' + size + ')';
        showToast('頭のサイズを変更: ' + size);
    });
});

// ============================================================================
// Low Latency Toggle
// ============================================================================
lowLatencyToggle.addEventListener('change', (e) => {
    if (e.target.checked) {
        processingMode.textContent = 'Low Latency';
        showToast('低遅延モードを有効化');
    } else {
        processingMode.textContent = 'Standard';
        showToast('低遅延モードを無効化');
    }
});

// ============================================================================
// Toast Notification
// ============================================================================
function showToast(text) {
    const toast = document.createElement('div');
    toast.textContent = text;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        background: rgba(0, 255, 136, 0.2);
        border: 1px solid #00ff88;
        border-radius: 4px;
        color: #00ff88;
        font-size: 14px;
        z-index: 10000;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        animation: slideIn 0.3s ease-out;
    `;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
