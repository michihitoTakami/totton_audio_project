// EQ Page Interactive Features

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const searchInput = document.getElementById('headphoneSearch');
    const searchResults = document.getElementById('searchResults');
    const profileName = document.getElementById('profileName');
    const preampSlider = document.getElementById('preampSlider');
    const preampValue = document.getElementById('preampValue');

    // Search functionality
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim().toLowerCase();
        if (query.length > 0) {
            searchResults.style.display = 'block';
            // Filter results
            const items = searchResults.querySelectorAll('.search-result-item');
            items.forEach(item => {
                const name = item.dataset.name.toLowerCase();
                if (name.includes(query)) {
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            });
        } else {
            searchResults.style.display = 'none';
        }
    });

    // Search result selection
    searchResults.addEventListener('click', (e) => {
        const item = e.target.closest('.search-result-item');
        if (item) {
            const name = item.dataset.name;
            const author = item.dataset.author;
            profileName.textContent = name;
            searchInput.value = name;
            searchResults.style.display = 'none';
            showToast(`プロファイルを選択: ${name}`);
        }
    });

    // Hide search results when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.style.display = 'none';
        }
    });

    // Preamp slider
    preampSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value).toFixed(1);
        preampValue.textContent = `${value} dB`;
    });

    // Action buttons
    const actionButtons = document.querySelectorAll('.action-btn');
    actionButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const btnText = btn.querySelector('.btn-text').textContent;
            if (btnText === '適用') {
                showToast('EQ設定を適用しました');
            } else if (btnText === 'コピー') {
                showToast('EQ設定をクリップボードにコピーしました');
            } else if (btnText === '無効化') {
                showToast('EQ設定を無効化しました');
            }
        });
    });

    // Toast notification
    function showToast(message) {
        const existingToast = document.querySelector('.toast');
        if (existingToast) {
            existingToast.remove();
        }

        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('show');
        }, 10);

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
});
