# OTA アップデート設計

## 概要

Totton AudioはWeb UI経由でファームウェアアップデートを提供します。ユーザーは管理画面から簡単にアップデートを実行できます。

---

## アップデート方式

### アプリケーション更新（推奨）

| 項目 | 内容 |
|------|------|
| 対象 | gpu_upsampler, Web UI, 設定ファイル |
| ダウンタイム | ~10秒 |
| ロールバック | 手動（前バージョンパッケージ保持） |
| 方式 | パッケージ差し替え + サービス再起動 |

---

## アップデートフロー

```
┌─────────────────────────────────────────────────────────────┐
│                     Web UI                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  現在のバージョン: 1.0.0                             │    │
│  │  最新バージョン:   1.1.0                             │    │
│  │                                                       │    │
│  │  [更新を確認] [アップデート適用]                      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
        │                           │
        │ 1. 確認                   │ 2. 適用
        ▼                           ▼
┌───────────────┐           ┌───────────────────────────────┐
│ Update Server │           │         Update Process         │
│               │           │                                │
│ /api/latest   │           │  1. パッケージダウンロード     │
│ /packages/    │           │  2. 整合性検証 (SHA256)        │
│               │           │  3. サービス停止               │
└───────────────┘           │  4. ファイル差し替え           │
                            │  5. サービス起動               │
                            │  6. ヘルスチェック             │
                            │  7. 完了通知                   │
                            └───────────────────────────────┘
```

---

## API設計

### 更新確認

```http
GET /api/update/check
```

**レスポンス**:
```json
{
  "current_version": "1.0.0",
  "latest_version": "1.1.0",
  "update_available": true,
  "release_notes": "バグ修正と性能改善",
  "package_size": 52428800,
  "release_date": "2025-12-01"
}
```

### 更新適用

```http
POST /api/update/apply
Content-Type: application/json

{
  "version": "1.1.0"
}
```

**レスポンス（進捗）**:
```json
{
  "status": "in_progress",
  "stage": "downloading",
  "progress": 45,
  "message": "パッケージをダウンロード中..."
}
```

### 更新ステータス

```http
GET /api/update/status
```

**レスポンス**:
```json
{
  "status": "completed",
  "previous_version": "1.0.0",
  "current_version": "1.1.0",
  "updated_at": "2025-12-01T10:30:00Z"
}
```

---

## パッケージ形式

### ディレクトリ構成

```
magicbox-update-1.1.0.tar.gz
├── manifest.json           # メタデータ
├── checksums.sha256        # ファイルハッシュ
├── bin/
│   └── gpu_upsampler_alsa  # メインバイナリ
├── lib/
│   └── *.so                # 共有ライブラリ
├── web/
│   └── ...                 # Web UI
├── data/
│   └── coefficients/       # フィルタ係数（変更時のみ）
└── scripts/
    ├── pre_update.sh       # 更新前スクリプト
    └── post_update.sh      # 更新後スクリプト
```

### manifest.json

```json
{
  "version": "1.1.0",
  "min_compatible": "1.0.0",
  "release_date": "2025-12-01",
  "release_notes": "バグ修正と性能改善",
  "components": [
    {
      "name": "gpu_upsampler_alsa",
      "path": "bin/gpu_upsampler_alsa",
      "checksum": "sha256:abc123..."
    },
    {
      "name": "web",
      "path": "web/",
      "checksum": "sha256:def456..."
    }
  ],
  "scripts": {
    "pre_update": "scripts/pre_update.sh",
    "post_update": "scripts/post_update.sh"
  }
}
```

---

## 更新スクリプト

### /usr/local/bin/magicbox-update

```bash
#!/bin/bash
#
# Totton Audio Update Script
#

set -euo pipefail

UPDATE_DIR="/opt/magicbox/updates"
INSTALL_DIR="/opt/magicbox"
BACKUP_DIR="/opt/magicbox/backup"
VERSION_FILE="/opt/magicbox/VERSION"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    logger -t "magicbox-update" "$*"
}

# 更新確認
check_update() {
    local server="${MAGICBOX_UPDATE_SERVER:-https://updates.magicbox.audio}"
    local current_version
    current_version=$(cat "$VERSION_FILE" 2>/dev/null || echo "0.0.0")

    log "Checking for updates... (current: $current_version)"

    local response
    response=$(curl -sf "${server}/api/latest?current=${current_version}" || echo "{}")

    echo "$response"
}

# パッケージダウンロード
download_package() {
    local version="$1"
    local server="${MAGICBOX_UPDATE_SERVER:-https://updates.magicbox.audio}"
    local package_url="${server}/packages/magicbox-update-${version}.tar.gz"
    local package_path="${UPDATE_DIR}/magicbox-update-${version}.tar.gz"

    log "Downloading update package: $version"

    mkdir -p "$UPDATE_DIR"
    curl -fL "$package_url" -o "$package_path"
    curl -fL "${package_url}.sha256" -o "${package_path}.sha256"

    # 検証
    log "Verifying package integrity..."
    cd "$UPDATE_DIR"
    sha256sum -c "${package_path}.sha256"

    echo "$package_path"
}

# バックアップ作成
create_backup() {
    local current_version
    current_version=$(cat "$VERSION_FILE" 2>/dev/null || echo "unknown")
    local backup_path="${BACKUP_DIR}/${current_version}-$(date +%Y%m%d%H%M%S)"

    log "Creating backup: $backup_path"

    mkdir -p "$backup_path"
    cp -a "${INSTALL_DIR}/bin" "$backup_path/"
    cp -a "${INSTALL_DIR}/web" "$backup_path/"
    cp -a "$VERSION_FILE" "$backup_path/" 2>/dev/null || true

    echo "$backup_path"
}

# 更新適用
apply_update() {
    local package_path="$1"
    local temp_dir
    temp_dir=$(mktemp -d)

    log "Extracting update package..."
    tar xzf "$package_path" -C "$temp_dir"

    # マニフェスト確認
    if [[ ! -f "${temp_dir}/manifest.json" ]]; then
        log "ERROR: Invalid update package (missing manifest)"
        return 1
    fi

    # pre_update スクリプト実行
    if [[ -f "${temp_dir}/scripts/pre_update.sh" ]]; then
        log "Running pre-update script..."
        bash "${temp_dir}/scripts/pre_update.sh"
    fi

    # サービス停止
    log "Stopping services..."
    systemctl stop magicbox-web gpu-upsampler || true

    # ファイル差し替え
    log "Installing new files..."
    [[ -d "${temp_dir}/bin" ]] && cp -a "${temp_dir}/bin/"* "${INSTALL_DIR}/bin/"
    [[ -d "${temp_dir}/lib" ]] && cp -a "${temp_dir}/lib/"* "${INSTALL_DIR}/lib/" 2>/dev/null || true
    [[ -d "${temp_dir}/web" ]] && cp -a "${temp_dir}/web/"* "${INSTALL_DIR}/web/"
    [[ -d "${temp_dir}/data" ]] && cp -a "${temp_dir}/data/"* "${INSTALL_DIR}/data/"

    # バージョン更新
    local new_version
    new_version=$(jq -r '.version' "${temp_dir}/manifest.json")
    echo "$new_version" > "$VERSION_FILE"

    # post_update スクリプト実行
    if [[ -f "${temp_dir}/scripts/post_update.sh" ]]; then
        log "Running post-update script..."
        bash "${temp_dir}/scripts/post_update.sh"
    fi

    # サービス起動
    log "Starting services..."
    systemctl start gpu-upsampler magicbox-web

    # クリーンアップ
    rm -rf "$temp_dir"
    rm -f "$package_path" "${package_path}.sha256"

    log "Update completed: $new_version"
}

# ヘルスチェック
health_check() {
    log "Running health check..."

    local retry=0
    local max_retry=30

    while [[ $retry -lt $max_retry ]]; do
        if curl -sf http://localhost/status > /dev/null; then
            log "Health check passed"
            return 0
        fi
        sleep 1
        ((retry++))
    done

    log "ERROR: Health check failed"
    return 1
}

# ロールバック
rollback() {
    local backup_path="$1"

    log "Rolling back to: $backup_path"

    systemctl stop magicbox-web gpu-upsampler || true

    cp -a "${backup_path}/bin/"* "${INSTALL_DIR}/bin/"
    cp -a "${backup_path}/web/"* "${INSTALL_DIR}/web/"
    [[ -f "${backup_path}/VERSION" ]] && cp "${backup_path}/VERSION" "$VERSION_FILE"

    systemctl start gpu-upsampler magicbox-web

    log "Rollback completed"
}

# メイン処理
case "${1:-check}" in
    check)
        check_update
        ;;
    download)
        download_package "${2:?Version required}"
        ;;
    apply)
        VERSION="${2:-}"
        if [[ -z "$VERSION" ]]; then
            # 最新バージョン取得
            VERSION=$(check_update | jq -r '.latest_version')
        fi

        BACKUP_PATH=$(create_backup)
        PACKAGE_PATH=$(download_package "$VERSION")

        if apply_update "$PACKAGE_PATH"; then
            if health_check; then
                log "Update successful"
            else
                log "Health check failed, rolling back..."
                rollback "$BACKUP_PATH"
                exit 1
            fi
        else
            log "Update failed, rolling back..."
            rollback "$BACKUP_PATH"
            exit 1
        fi
        ;;
    rollback)
        BACKUP_PATH="${2:?Backup path required}"
        rollback "$BACKUP_PATH"
        ;;
    *)
        echo "Usage: $0 {check|download <version>|apply [version]|rollback <backup_path>}"
        exit 1
        ;;
esac
```

---

## Web UI 統合

### FastAPI エンドポイント

```python
# web/routers/update.py
from fastapi import APIRouter, BackgroundTasks
import subprocess
import json

router = APIRouter(prefix="/api/update", tags=["update"])

@router.get("/check")
async def check_update():
    result = subprocess.run(
        ["/usr/local/bin/magicbox-update", "check"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

@router.post("/apply")
async def apply_update(version: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_update, version)
    return {"status": "started", "message": "Update started in background"}

def run_update(version: str):
    subprocess.run(
        ["/usr/local/bin/magicbox-update", "apply", version],
        capture_output=True
    )
```

---

## セキュリティ

### パッケージ署名（将来）

```bash
# 署名検証
gpg --verify magicbox-update-1.1.0.tar.gz.sig magicbox-update-1.1.0.tar.gz
```

### HTTPS必須

更新サーバとの通信は必ずHTTPS経由。

---

## 関連ドキュメント

- [image-build.md](./image-build.md) - イメージ作成
- [factory-provisioning.md](./factory-provisioning.md) - 工場プロビジョニング
- [../reliability/error-recovery.md](../reliability/error-recovery.md) - エラー回復
