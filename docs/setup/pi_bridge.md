# Raspberry Pi ブリッジ初期化ガイド

Jetsonを純粋なGPU DSPノードとして動かし、UAC2デバイスはRaspberry Piでエミュレートする構成のセットアップ手順です。  
PiはUSB 1本でPCに接続し、UAC2 + ECMコンポジットデバイスとして振る舞います。Pi内部ではPipeWireでPCMを受け取り、RTPでJetsonへ転送し、さらにnginxでJetson Web UI(:8000)をプロキシします。

```
PC ──USB (UAC2 + ECM)──> Raspberry Pi ──Ethernet──> Jetson Orin ──> USB DAC
            │                                    │
            └── Web UIアクセス (nginx → Jetson) ─┘
```

---

## 前提条件

- Raspberry Pi 4/5 (USB-C OTG対応) + Raspberry Pi OS (Debian Bookworm系)
- `dtoverlay=dwc2,dr_mode=peripheral` を `/boot/firmware/config.txt` (または `/boot/config.txt`) で有効化できること
- PiとJetsonを結ぶ有線LAN (例: `eth1` 同士をクロス/スイッチ接続)
- Jetsonが `192.168.56.2` などの固定IPで起動していること（値は後述のオプションで変更可）
- リポジトリを Pi 上に clone 済み (`/home/pi/michy_os` など)

> **注意**: スクリプトはPi上で実行します。全ての処理（パッケージインストール・systemd設定・nginx構成）が Pi で行われます。

---

## 1. スクリプト概要

`scripts/pi/setup-dev-bridge.sh` が以下を自動化します。

1. 必要パッケージ (`pipewire-bin`, `nginx`, `jq` など) のインストール
2. UAC2+ECM コンポジットガジェット (`magicbox-pi-gadget.service`) のデプロイ  
   - 32bit / 最大 768kHz までのサンプルレートを広告
3. `dhcpcd` 設定を書き換え、`usb0` (PC側) / `eth1`(Jetson側) に固定IPを割り当て
4. PipeWire RTP送出スクリプト (`/usr/local/bin/magicbox-rtp-forward.sh`) と systemd サービス (`magicbox-rtp-forward.service`) を作成
5. nginxを設定し、Pi上の `http://<pi>:80/` を Jetson の `http://<jetson>:8000/` にプロキシ

アンインストールも同スクリプトで行えます。

---

## 2. インストール手順

```bash
cd /path/to/michy_os
sudo ./scripts/pi/setup-dev-bridge.sh install \
  --pi-user pi \
  --pc-subnet 192.168.55.1/24 \
  --jetson-if eth1 \
  --pi-jetson-ip 192.168.56.1/24 \
  --jetson-ip 192.168.56.2 \
  --rtp-port 6000 \
  --rtp-rate 768000 \
  --nginx-listen 80 \
  --jetson-web-port 8000
```

引数を省略すると上記のデフォルトが適用されます。  
完了メッセージが出たら **必ずPiを再起動** し、`usb0` がPC側に正しく認識されることを確認してください。

### 2.1 生成される主要ファイル

| パス | 用途 |
|------|------|
| `/usr/local/bin/magicbox-pi-gadget` | UAC2+ECMガジェット制御スクリプト |
| `/etc/systemd/system/magicbox-pi-gadget.service` | ガジェット起動ワンサービス |
| `/usr/local/bin/magicbox-rtp-forward.sh` | PipeWire → RTP転送ロジック |
| `/etc/systemd/system/magicbox-rtp-forward.service` | PipeWire連携サービス (User=`pi`) |
| `/etc/nginx/sites-available/magicbox` | nginx リバプロ設定 |
| `/etc/dhcpcd.conf` | `usb0` / `eth1` の固定IP設定 (マーカー付き) |

---

## 3. 動作確認

1. **USBガジェット**  
   ```bash
   sudo systemctl status magicbox-pi-gadget.service
   sudo /usr/local/bin/magicbox-pi-gadget status
   ```
   PC側で `lsusb` → 「Magic Box Pi Bridge」が見えること、`usb0` に `192.168.55.1/24` が付与されていることを確認。

2. **RTP転送**  
   ```bash
   sudo systemctl status magicbox-rtp-forward.service
   ```
   `journalctl -u magicbox-rtp-forward` でエラーが無いか確認。Jetson側で `pw-top` や `tcpdump udp port 6000` などで受信できているか確認します。

3. **Web UI**  
   PCのブラウザから `http://192.168.55.1/` (またはPiのmDNS名) にアクセスし、Jetson Web UIが表示されればOK。

---

## 4. アンインストール

```bash
sudo ./scripts/pi/setup-dev-bridge.sh uninstall
```

- `magicbox-pi-gadget.service` / `magicbox-rtp-forward.service` 無効化・削除
- `dhcpcd.conf` の追記ブロック削除
- nginx サイトの削除
- `/usr/local/bin/magicbox-*.sh` 削除

必要に応じて `/boot/firmware/config.txt` の `dtoverlay=dwc2,...` を手動で戻してください。

---

## 5. オプション一覧

| 引数 | 内容 | 既定値 |
|------|------|--------|
| `--pi-user` | PipeWire と nginx を動かすユーザー | `pi` |
| `--pc-subnet` | `usb0` に割り当てる CIDR | `192.168.55.1/24` |
| `--jetson-if` | Jetsonへ繋ぐインターフェース名 | `eth1` |
| `--pi-jetson-ip` | Pi側のJetsonリンクIP | `192.168.56.1/24` |
| `--jetson-ip` | JetsonのIP (RTP / Web UI) | `192.168.56.2` |
| `--rtp-port` | RTP送出UDPポート | `6000` |
| `--rtp-rate` | PipeWire RTPサンプルレート | `768000` |
| `--nginx-listen` | Piの公開ポート | `80` |
| `--jetson-web-port` | JetsonのWeb UIポート | `8000` |
| `--sample-rates` | UAC2が広告するサンプルレート (CSV) | `44100,...,768000` |

環境変数 `MAGICBOX_*` を使えば `.bashrc` などから一括設定も可能です（スクリプト内で同名の変数を参照します）。

---

## 6. よくある質問

### Q. Jetson側の設定は？
- JetsonはRTP受信（PipeWire `libpipewire-module-rtp-session` など）を有効化し、`gpu_upsampler_alsa` の入力を `pipewire` PCMへ切り替えます。  
- Web UIは引き続きJetson上で稼働し、PiのnginxがHTTPを中継します。

### Q. 768kHz/32bitのストリームはPiで耐えられる？
- 帯域は約 50 Mbps 程度なので USB 2.0 HS と Gigabit Ethernet の範囲内です。
- `pw-loopback`・PipeWireのバッファを適切に取り、`dmesg` や `journalctl` でXRUNが出ていないか確認してください。

### Q. USBとLANの両方で同時通信できる？
- 本スクリプトは UAC2 (isochronous) と ECM (bulk) を同一コンポジットに含めています。ホストPCではUSBオーディオ＋USBイーサ双方が同時に利用できます。

---

## 7. 既知の制限

- PipeWireノード名の検出は `alsa_input.usb-Magic_Box` パターンで行っており、製品名を変更した場合は `CAPTURE_PATTERN` 環境変数で調整が必要です。
- `systemd-networkd` ではなく `dhcpcd` に固定IPを記述しています。他のネットワークマネージャと併用する場合は競合に注意してください。
- `dtoverlay=dwc2` の反映には再起動が必須です。初回セットアップ後は必ずPiを再起動してください。

---

以上で、ラズパイを中継ノードとして使う際の自動初期化が完了します。Jetson側のRTP受信/サービス構成と合わせて、PC→Pi→Jetsonのシームレスなオーディオ経路を構築してください。

