// EQ Page Interactive Features

document.addEventListener('DOMContentLoaded', () => {
    const deployBtn = document.getElementById('deployBtn');
    const copyBtn = document.getElementById('copyBtn');
    const eqProfile = document.getElementById('eqProfile');

    // Deploy button
    deployBtn.addEventListener('click', () => {
        const profile = eqProfile.value.trim();
        if (profile) {
            showToast('EQプロファイルをデプロイしました');
        } else {
            showToast('EQプロファイルが空です', 'warning');
        }
    });

    // Copy button
    copyBtn.addEventListener('click', async () => {
        const profile = eqProfile.value.trim();
        if (profile) {
            try {
                await navigator.clipboard.writeText(profile);
                showToast('クリップボードにコピーしました');
            } catch (err) {
                showToast('コピーに失敗しました', 'error');
            }
        } else {
            showToast('コピーする内容がありません', 'warning');
        }
    });

    // Toast notification
    function showToast(message, type = 'success') {
        const existingToast = document.querySelector('.toast');
        if (existingToast) {
            existingToast.remove();
        }

        const toast = document.createElement('div');
        toast.className = 'toast';
        if (type === 'warning') {
            toast.style.background = 'rgba(255, 170, 0, 0.9)';
        } else if (type === 'error') {
            toast.style.background = 'rgba(255, 68, 68, 0.9)';
        }
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
