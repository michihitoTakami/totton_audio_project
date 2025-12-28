# Jetson 評価者向け導入ガイド（ソース不要 / Docker）

Issue: #1060（Epic: #1051）

このドキュメントは **Jetson 上で評価版を最短で起動する** ための手順です（ソースコード不要）。

---

## 前提（必須）

- **JetPack 6.1 以降**（L4T r36.4+）
- **Docker Engine + Docker Compose v2**
- **NVIDIA Container Runtime が有効**
- **音声デバイスをコンテナに渡せる**こと（`/dev/snd` が存在する）
- インターネット接続（GHCR から image を pull するため）

確認コマンド（例）:

```bash
# JetPack / L4T
cat /etc/nv_tegra_release || true

# Docker / Compose
docker version
docker compose version

# NVIDIA runtime（nvidia-smi が通ることを目安にする）
sudo docker run --rm --runtime=nvidia nvidia-smi

# ALSA デバイス
ls -l /dev/snd || true
```

---

## 起動手順（一本道 / runtime-only）

配布物（またはリポジトリ）に含まれる `docker/` ディレクトリを Jetson に配置した前提です。

```bash
cd docker

# 1) 起動（ソース不要: GHCR image を pull して起動）
docker compose -f jetson/docker-compose.jetson.runtime.yml up -d

# 2) ログ追跡
docker compose -f jetson/docker-compose.jetson.runtime.yml logs -f
```

Web UI:
- `http://<JetsonのIP>/`（runtime compose は `80:80` を公開します）

停止:

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.runtime.yml down
```

---

## ログ取得（不具合報告用）

```bash
cd docker

# コンテナログ（直近1時間）
docker compose -f jetson/docker-compose.jetson.runtime.yml logs --since 1h --no-color > magicbox.log

# 追加の環境情報
{
  echo "=== uname ==="
  uname -a
  echo
  echo "=== nv_tegra_release ==="
  cat /etc/nv_tegra_release 2>/dev/null || true
  echo
  echo "=== docker info (nvidia) ==="
  docker info 2>/dev/null | grep -iE "nvidia|runtime" || true
  echo
  echo "=== docker compose ps ==="
  docker compose -f jetson/docker-compose.jetson.runtime.yml ps
} > env.txt

# コンテナ内の ALSA デバイス一覧（参考）
docker compose -f jetson/docker-compose.jetson.runtime.yml exec -T magicbox aplay -l || true

# まとめて圧縮
tar czf magicbox-debug-$(date +%Y%m%d_%H%M%S).tar.gz magicbox.log env.txt
```

---

## 既知トラブル

### NVIDIA runtime が無効（起動できない / GPUが見えない）

症状例:
- `unknown runtime specified nvidia`
- GPU を使う処理が失敗する

確認:

```bash
sudo docker run --rm --runtime=nvidia nvidia-smi
```

対処:
- JetPack のインストール状態を確認し、NVIDIA Container Runtime を有効化してください。

### 音が出ない（/dev/snd が見えない）

症状例:
- ALSA device open に失敗
- コンテナは起動するが再生できない

確認:

```bash
ls -l /dev/snd
cd docker
docker compose -f jetson/docker-compose.jetson.runtime.yml exec -T magicbox aplay -l
```

対処:
- DAC/Loopback の接続・認識（ホスト側）を確認し、`/dev/snd` が存在する状態にしてください。

### Web UI にアクセスできない

確認:

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.runtime.yml ps
curl -fsS http://127.0.0.1/ >/dev/null && echo OK
```

対処:
- Jetson の IP を確認し、`http://<JetsonのIP>/` にアクセスしてください。
- ファイアウォール/ルータ設定で TCP/80 が遮断されていないか確認してください。

---

## ロールバック / 初期化

### 1) 単純停止（設定は保持）

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.runtime.yml down
```

### 2) 設定も含めて完全初期化（named volume も削除）

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.runtime.yml down -v
```

### 3) 以前のバージョンへ戻す（imageタグ固定）

```bash
cd docker

# 例: 既知のタグへ固定して起動
MAGICBOX_IMAGE=ghcr.io/michihitotakami/totton-audio-system:<tag> \
  docker compose -f jetson/docker-compose.jetson.runtime.yml up -d
```
