# Jetson 開発用自動起動スクリプト

Jetson Orin上で Magic Box を開発する際、再起動の度に
`gpu_upsampler_alsa` と Web UI を手動起動する手間をなくすための
systemd セットアップスクリプトを追加しました。

> ⚠️ **注意**: これは `/opt/magicbox` 以下に正規インストールされた
> プロダクション環境向けではなく、
> 「GitHub から clone した開発用ワークツリー」専用の機能です。

---

## スクリプト概要

`scripts/jetson/setup-dev-autostart.sh` が以下を行います。

1. リポジトリルート（デフォルト: スクリプト位置から逆算）を検出
2. `gpu-upsampler-dev.service` と `magicbox-web-dev.service` を `/etc/systemd/system/` に生成
3. `systemctl enable --now` で再起動後も常駐するように設定
4. `uninstall` でサービス停止＋ファイル削除、`status` で状態参照が可能

---

## 依存条件

- Jetson Orin 上で本リポジトリを直接 clone 済みであること
- `cmake -B build && cmake --build build` で `build/gpu_upsampler_alsa` を生成済み
- `uv` コマンドがインストール済み（`uv run uvicorn ...` を systemd で利用）
- `sudo` 実行権限（サービスは system レベルで登録）

---

## 使い方

### インストール

```bash
sudo ./scripts/jetson/setup-dev-autostart.sh install \
  --user jetson \
  --port 80
```

- `--user` : Web UI を動かす Linux ユーザー（デフォルト `jetson`）
- `--port` : Web UI ポート。80 未満を指定すると
  `CAP_NET_BIND_SERVICE` を自動付与します（既定は 80）。
- `--repo` : リポジトリルートを手動指定したい場合に利用

インストール後の確認:

```bash
sudo systemctl status gpu-upsampler-dev.service magicbox-web-dev.service
```

### 解除

```bash
sudo ./scripts/jetson/setup-dev-autostart.sh uninstall
```

`systemctl disable --now` を実行した上でサービスファイルを削除します。

### 状態表示

```bash
sudo ./scripts/jetson/setup-dev-autostart.sh status
```

内部的に `systemctl status` を呼び出します。失敗時には systemd の
標準的なエラーをそのまま確認できます。

---

## 作成される systemd サービス

| サービス名 | 役割 | 主な設定 |
|------------|------|----------|
| `gpu-upsampler-dev.service` | `build/gpu_upsampler_alsa` をリポジトリ root で実行 | `LimitRTPRIO=99`, `LimitMEMLOCK=infinity`, `Restart=always` |
| `magicbox-web-dev.service` | `uv run uvicorn web.main:app` を起動し Web UI を提供 | `After=gpu-upsampler-dev`, `Restart=always`, ポート <1024 の場合は Capability 付与 |

- どちらも `WantedBy=multi-user.target` のため、Jetson 再起動後に自動で起動します。
- Web UI は指定ユーザー権限で実行され、`PartOf=gpu-upsampler-dev` により
  音声デーモンの停止に連動して終了します。

---

## トラブルシューティング

| 症状 | 対応 |
|------|------|
| `uv` が見つからない | `curl -LsSf https://astral.sh/uv/install.sh \| sh` などでインストール |
| `build/gpu_upsampler_alsa` が無い | Jetson 上で `cmake -B build && cmake --build build -j$(nproc)` を実行 |
| ポート 80 で起動に失敗する | `--port 11881` など 1024 以上を指定して動作確認 |
| サービスを完全に無効化したい | `sudo systemctl disable --now magicbox-web-dev gpu-upsampler-dev` を実行 |

---

## 今後の拡張アイデア

- `magicbox-gadget.service` など USB Gadget まわりとの連携
- `/opt/magicbox` パッケージとの差分を自動検出し警告
- `--oneshot` でインストールのみ／有効化のみを切り替えるオプション


