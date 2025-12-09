# Journald ログローテーション運用ガイド

## 目的
- journald ログが肥大化して NVMe / microSD を圧迫しないよう、容量・保持期間・レートリミットを統一する
- 反映手順と確認コマンドを明文化し、各サービス（gpu-upsampler / magicbox-web / magicbox-gadget / Jetson常駐ツール）で共通運用できるようにする

## 推奨設定値（/etc/systemd/journald.conf.d/magicbox.conf）
リポジトリ同梱のテンプレート `systemd/journald.conf.d/magicbox.conf` をそのまま配置する。

| 項目 | 値 | 理由 |
|------|----|------|
| Storage | persistent | 再起動後も直近の障害解析ができる |
| Compress | yes | 容量を抑えつつ保持期間を確保 |
| SystemMaxUse | 200M | 1TB SSDでも0.1%未満に制限し、肥大化を防止 |
| SystemKeepFree | 512M | 低空き容量時でも最低限の空きを確保 |
| SystemMaxFileSize | 16M | 一度のスパイクで巨大ファイルにならないよう制限 |
| MaxRetentionSec | 14day | 2週間の障害追跡を確保 |
| RateLimitIntervalSec | 30s | ログスパムを時間で平滑化 |
| RateLimitBurst | 500 | burstを許容しつつ無限出力を防ぐ |
| ForwardToSyslog | no | 外部syslog未使用のため重複出力を防止 |

## 配置・反映手順
```bash
# 1) 設定を配置（初回はディレクトリを作る）
sudo install -Dm644 systemd/journald.conf.d/magicbox.conf \
  /etc/systemd/journald.conf.d/magicbox.conf

# 2) 再読み込み
sudo systemctl restart systemd-journald.service

# 3) 反映確認
systemctl status systemd-journald --no-pager
journalctl --disk-usage
journalctl --header --system | head -n 15
```

## 運用・定期確認
- ディスク使用量: `journalctl --disk-usage` （200Mを超えないことを確認）
- 保持期間: `journalctl --since "14 days ago" --until "5 minutes ago" -u 'magicbox*' -u gpu-upsampler | tail`
- 再起動後の永続化確認: `sudo ls -lh /var/log/journal/*`
- エラー集中の検知: `journalctl -p err..alert --since "1 hour ago" -u 'magicbox*' -u gpu-upsampler`

### 手動ローテーション・容量逼迫時の対応
```bash
# 容量で縮小（150Mまで即時vacuum）
sudo journalctl --vacuum-size=150M

# 期間で縮小（7日より古いものを削除）
sudo journalctl --vacuum-time=7d

# ログ破損チェック
sudo journalctl --verify
```

### サービス別クイックビュー
- メイン（CUDA/ALSA）: `journalctl -u gpu-upsampler -b --no-pager`
- Web UI: `journalctl -u magicbox-web -b --no-pager`
- USB Gadget: `journalctl -u magicbox-gadget -b --no-pager`
- Docker併用時: Dockerは `json-file` ローテーション（10m×3ファイル）を継続利用。journaldはホスト側のsystemdサービス向け。

## 受け入れ条件との対応
- 推奨値と理由を上表で明文化
- 配置・再起動・反映確認の具体コマンドを提示
- ローテーション状況確認と容量逼迫時の対応コマンドを提示
