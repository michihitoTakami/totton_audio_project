# トラブルシューティング

## 概要

Totton Audioで発生する可能性のある問題と解決方法をまとめています。

---

## クイック診断

```bash
# 全サービスステータス確認
systemctl status magicbox-gadget gpu-upsampler magicbox-web

# 直近のエラーログ
journalctl -p err --since "1 hour ago"

# USB Gadget状態
/usr/local/bin/magicbox-gadget-setup status

# ネットワーク状態
networkctl status usb0
```

---

## 1. USB関連の問題

### 1.1 PCがTotton Audioを認識しない

**症状**: USBケーブルを接続してもPCにデバイスが表示されない

**確認項目**:
```bash
# UDC（USB Device Controller）確認
ls /sys/class/udc/

# Gadget設定確認
cat /sys/kernel/config/usb_gadget/magicbox/UDC

# dmesgでUSBエラー確認
dmesg | grep -i usb | tail -20
```

**解決策**:

1. USBケーブルを交換（データ対応ケーブルか確認）
2. 別のUSBポートを試す
3. Gadgetサービス再起動:
   ```bash
   sudo systemctl restart magicbox-gadget
   ```
4. Jetson再起動

---

### 1.2 オーディオデバイスとして認識されない

**症状**: USB Ethernetは認識されるが、オーディオデバイスが見えない

**確認項目**:
```bash
# カーネルモジュール確認
lsmod | grep uac2

# UAC2関数確認
ls /sys/kernel/config/usb_gadget/magicbox/functions/uac2.usb0/
```

**解決策**:

1. UAC2モジュールロード:
   ```bash
   sudo modprobe usb_f_uac2
   ```
2. モジュールが存在しない場合: [kernel-modules.md](./usb-gadget/kernel-modules.md) 参照

---

### 1.3 Windowsで認識されない

**症状**: Linux/macOSでは動作するがWindowsで認識されない

**確認項目**:
- Windows 10 バージョン 1703以降か
- デバイスマネージャーで不明なデバイスがないか

**解決策**:

1. Windowsを最新版に更新
2. ECM → RNDIS に変更（[composite-gadget.md](./usb-gadget/composite-gadget.md) 参照）
3. Bonjour Print Servicesをインストール（mDNS用）

---

## 2. ネットワーク関連の問題

### 2.1 Web UIにアクセスできない

**症状**: `http://192.168.55.1/` にアクセスできない

**確認項目**:
```bash
# Jetson側
ip addr show usb0
systemctl status magicbox-web

# PC側
ping 192.168.55.1
```

**解決策**:

1. IPアドレス手動設定（PC側）:
   ```bash
   # Linux
   sudo ip addr add 192.168.55.100/24 dev usb0
   ```

2. Webサービス再起動:
   ```bash
   sudo systemctl restart magicbox-web
   ```

---

### 2.2 magicbox.local が解決できない

**症状**: IPアドレスでは接続できるが、ホスト名では接続できない

**確認項目**:
```bash
# Avahi確認
systemctl status avahi-daemon
avahi-browse -all
```

**解決策**:

1. Avahi再起動:
   ```bash
   sudo systemctl restart avahi-daemon
   ```

2. IPアドレス直接アクセス: `http://192.168.55.1/`

3. Windows: Bonjour Print Servicesインストール

---

## 3. オーディオ関連の問題

### 3.1 音が出ない

**症状**: PCから再生しているが音声が出力されない

**確認項目**:
```bash
# デーモンステータス
systemctl status gpu-upsampler

# ALSAデバイス
aplay -l

# 入力検知
journalctl -u gpu-upsampler | grep -i "input"
```

**解決策**:

1. PC側の出力デバイスがTotton Audioになっているか確認
2. DAC接続確認
3. デーモン再起動:
   ```bash
   sudo systemctl restart gpu-upsampler
   ```

---

### 3.2 ノイズ・クラック音

**症状**: 音は出るがノイズやクラック音が発生する

**確認項目**:
```bash
# XRUNカウント
journalctl -u gpu-upsampler | grep -i "xrun"

# GPU使用率
nvidia-smi

# バッファ状態
cat /tmp/gpu_upsampler_stats.json
```

**解決策**:

1. バッファサイズ増加（config.json）:
   ```json
   {
     "bufferSize": 524288,
     "periodSize": 65536
   }
   ```

2. GPU負荷が高い場合はフォールバックモード確認

3. USBケーブル品質確認

---

### 3.3 レート切り替え時のポップノイズ

**症状**: 44.1kHz⇔48kHz切り替え時にポップ音

**確認項目**:
```bash
# Soft Mute設定確認
journalctl -u gpu-upsampler | grep -i "mute"
```

**解決策**:

1. Soft Mute有効確認（通常はデフォルト有効）
2. デーモン再起動で改善する場合あり

---

## 4. サービス関連の問題

### 4.1 サービスが起動しない

**症状**: `systemctl start gpu-upsampler` でエラー

**確認項目**:
```bash
# 詳細ステータス
systemctl status gpu-upsampler -l

# ジャーナル
journalctl -xeu gpu-upsampler
```

**解決策**:

1. 依存サービス確認:
   ```bash
   systemctl status magicbox-gadget
   ```

2. GPU確認:
   ```bash
   nvidia-smi
   ls /dev/nvidia*
   ```

3. 設定ファイル確認:
   ```bash
   cat /opt/magicbox/config.json | jq .
   ```

---

### 4.2 Watchdogタイムアウト

**症状**: サービスが自動再起動を繰り返す

**確認項目**:
```bash
# Watchdog状態
systemctl show gpu-upsampler --property=WatchdogTimestamp

# コアダンプ
coredumpctl list
```

**解決策**:

1. メモリ不足確認:
   ```bash
   free -h
   ```

2. GPU問題確認:
   ```bash
   nvidia-smi
   ```

3. ログで原因特定後、設定調整

---

## 5. 性能関連の問題

### 5.1 処理が追いつかない

**症状**: XRUNが頻発、音切れ

**確認項目**:
```bash
# GPU使用率
nvidia-smi dmon -s u

# メモリ使用率
free -h

# CPU使用率
top -p $(pgrep gpu_upsampler)
```

**解決策**:

1. 他のGPUプロセス終了
2. フィルタタップ数削減（将来対応）
3. バッファサイズ増加

---

### 5.2 起動が遅い

**症状**: 電源ON後、Ready状態になるまで1分以上かかる

**確認項目**:
```bash
# 起動時間分析
systemd-analyze blame | head -20

# サービス別時間
systemd-analyze critical-chain gpu-upsampler
```

**解決策**:

1. フィルタ係数プリロード確認
2. 不要サービス無効化
3. NVMe性能確認

---

## 6. アップデート関連の問題

### 6.1 アップデートが失敗する

**症状**: Web UIからのアップデートがエラー

**確認項目**:
```bash
# 更新ログ
journalctl -u magicbox-update

# ディスク容量
df -h
```

**解決策**:

1. インターネット接続確認（アップデートサーバへ）
2. ディスク容量確保
3. 手動アップデート:
   ```bash
   sudo /usr/local/bin/magicbox-update apply
   ```

---

## 7. ログ収集

問題報告時に収集すべきログ:

```bash
# システム情報
uname -a
cat /etc/os-release
nvidia-smi

# サービスログ
journalctl -u magicbox-gadget --since "1 hour ago" > gadget.log
journalctl -u gpu-upsampler --since "1 hour ago" > upsampler.log
journalctl -u magicbox-web --since "1 hour ago" > web.log

# 設定
cp /opt/magicbox/config.json config.json

# 統計
cp /tmp/gpu_upsampler_stats.json stats.json

# まとめて圧縮
tar czf magicbox-debug-$(date +%Y%m%d).tar.gz *.log *.json
```

---

## サポート

- **GitHub Issues**: https://github.com/michihitoTakami/totton_audio/issues
- **ドキュメント**: https://github.com/michihitoTakami/totton_audio/docs/jetson/

---

## 関連ドキュメント

- [quality/test-checklist.md](./quality/test-checklist.md) - テストチェックリスト
- [reliability/error-recovery.md](./reliability/error-recovery.md) - エラー回復
- [systemd/service-design.md](./systemd/service-design.md) - サービス設計
