# Web UI セットアップガイド

Magic Box Project の Web コントロールインターフェースの設定と使用方法。

## 前提条件

- [クイックスタート](quick_start.md) を完了していること
- Python 3.11以上、uv がインストール済み

## 概要

Web UI は FastAPI ベースで、2つのページを提供:

| ページ | URL | 用途 |
|--------|-----|------|
| User Page | `/` | デバイス選択、EQ設定 |
| Admin Page | `/admin` | Daemon制御、統計監視 |

## 起動方法

### 開発環境

```bash
# 依存関係インストール
uv sync

# サーバー起動 (ポート 11881)
uv run python web/main.py
```

### systemd (本番環境)

```bash
# サービスファイルをコピー
sudo cp systemd/gpu_upsampler_web.service /etc/systemd/system/

# 有効化・起動
sudo systemctl daemon-reload
sudo systemctl enable gpu_upsampler_web
sudo systemctl start gpu_upsampler_web
```

## アクセス方法

### ローカルアクセス

```
http://127.0.0.1:11881/       # ユーザーページ
http://127.0.0.1:11881/admin  # 管理者ページ
```

### 外部アクセス (LAN内)

デフォルトでは `127.0.0.1` にバインドされているため、外部からはアクセス不可。

外部アクセスを許可する場合は `web/main.py` の最終行を変更:

```python
# Before (localhost only)
uvicorn.run(app, host="127.0.0.1", port=11881)

# After (all interfaces)
uvicorn.run(app, host="0.0.0.0", port=11881)
```

その後、`http://<IP_ADDRESS>:11881/` でアクセス可能。

## 認証の追加 (推奨)

現在の実装には認証がありません。外部公開する場合は以下の方法で保護してください。

### 方法1: Nginx リバースプロキシ + Basic認証

```bash
# Nginx インストール
sudo apt install nginx apache2-utils

# パスワードファイル作成
sudo htpasswd -c /etc/nginx/.htpasswd admin
```

`/etc/nginx/sites-available/gpu_upsampler`:
```nginx
server {
    listen 80;
    server_name gpu-upsampler.local;

    location / {
        auth_basic "GPU Upsampler";
        auth_basic_user_file /etc/nginx/.htpasswd;

        proxy_pass http://127.0.0.1:11881;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# 有効化
sudo ln -s /etc/nginx/sites-available/gpu_upsampler /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 方法2: FastAPI 内蔵 Basic認証

`web/main.py` に追加:

```python
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

# 環境変数または設定ファイルから読み込み推奨
ADMIN_USER = "admin"
ADMIN_PASS = "your-secure-password"

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, ADMIN_USER)
    correct_pass = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# 管理者ページに認証を追加
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(username: str = Depends(verify_credentials)):
    return get_admin_html()
```

## API エンドポイント一覧

### ステータス
| Method | Endpoint | 説明 |
|--------|----------|------|
| GET | `/status` | 全体ステータス取得 |
| GET | `/daemon/status` | Daemon詳細ステータス |
| GET | `/devices` | ALSAデバイス一覧 |

### Daemon制御
| Method | Endpoint | 説明 |
|--------|----------|------|
| POST | `/daemon/start` | Daemon起動 |
| POST | `/daemon/stop` | Daemon停止 |
| POST | `/daemon/restart` | Daemon再起動 |
| POST | `/restart` | 設定リロード (SIGHUP) |

### 設定
| Method | Endpoint | 説明 |
|--------|----------|------|
| POST | `/settings` | 設定更新 |

### EQ管理
| Method | Endpoint | 説明 |
|--------|----------|------|
| GET | `/eq/profiles` | プロファイル一覧 |
| GET | `/eq/active` | アクティブなEQ取得 |
| POST | `/eq/activate/{name}` | EQ有効化 |
| POST | `/eq/deactivate` | EQ無効化 |
| POST | `/eq/import` | プロファイルインポート |
| DELETE | `/eq/profiles/{name}` | プロファイル削除 |

## ファイアウォール設定

UFW を使用している場合:

```bash
# LAN内のみ許可 (例: 192.168.1.0/24)
sudo ufw allow from 192.168.1.0/24 to any port 11881

# または特定IPのみ
sudo ufw allow from 192.168.1.100 to any port 11881
```

## トラブルシューティング

### サーバーが起動しない

```bash
# ポート競合確認
ss -tlnp | grep 11881

# 依存関係確認
uv sync
```

### 外部からアクセスできない

1. `host="0.0.0.0"` に変更済みか確認
2. ファイアウォールでポート開放済みか確認
3. 同一ネットワーク上にいるか確認

### Daemon制御が動作しない

```bash
# バイナリ存在確認
ls -la build/gpu_upsampler_alsa

# 権限確認 (バイナリ実行権限)
chmod +x build/gpu_upsampler_alsa
```

## 関連ドキュメント

- [PC開発環境](pc_development.md) - デーモン起動・ALSA/TCP設定
- [REST API仕様](../api/README.md) - API詳細ドキュメント
- [ビルド手順](build.md) - C++バイナリのビルド
