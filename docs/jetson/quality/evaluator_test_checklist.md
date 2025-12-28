# 評価者向けテストチェックリスト（Trial）

Issue: #1063（Epic: #1051）

目的: 評価者が「音が出る/レート切替/EQ/ログ」までを迷わず確認できること。

---

## 0. 事前情報（記録しておく）

- [ ] Jetson 機種（例: Orin Nano Super 8GB）/ JetPack バージョン（`cat /etc/nv_tegra_release`）
- [ ] DAC 名称 / 接続方式（USB）
- [ ] 入力経路（USB gadget / RTP）
- [ ] Raspberry Pi を使う場合: 機種 / OS / キャプチャデバイス

---

## 1. 起動確認（Jetson）

- [ ] `docker compose -f docker/jetson/docker-compose.jetson.runtime.yml ps` で `magicbox` が Up
- [ ] Web UI が開ける（デフォルト: `http://192.168.55.1/`）
- [ ] コンテナログに致命的エラーが出ていない

---

## 2. 音が出る（最重要）

- [ ] 無音にならない
- [ ] 片chにならない
- [ ] 明らかな歪み/クリック/周期ノイズがない
- [ ] 1分以上連続再生できる

---

## 3. レート切替（44.1k 系 / 48k 系）

入力側（PC or Pi）でサンプルレートを切り替え、挙動を確認します。

- [ ] 44.1kHz 系 → 48kHz 系へ切り替えても継続再生できる（短いミュートは許容）
- [ ] 48kHz 系 → 44.1kHz 系へ戻しても継続再生できる
- [ ] 切り替え時に過大なポップノイズが出ない

（RTP入力の場合）
- [ ] Pi 側でレート変更後、送信が自動追従する（必要ならパイプライン再起動される）
- [ ] Jetson 側が受信を継続する（UDP 46000-46002）

---

## 4. EQ の適用/解除

- [ ] Web UI から任意の EQ を適用できる
- [ ] 音の変化が体感できる（極端な破綻がない）
- [ ] EQ を解除/別プロファイルへ切替できる

---

## 5. ログ取得（不具合報告用）

少なくとも以下を採取できること。

### Jetson

- [ ] `docker compose -f docker/jetson/docker-compose.jetson.runtime.yml logs --since 1h --no-color > magicbox.log`
- [ ] `docker compose -f docker/jetson/docker-compose.jetson.runtime.yml ps` の結果
- [ ] `cat /etc/nv_tegra_release` の結果

### Raspberry Pi（使用時）

- [ ] compose 利用なら `docker compose -f raspberry_pi/docker-compose.raspberry_pi.runtime.yml logs --since 1h --no-color > raspi.log`
- [ ] `rtp_sender` 利用なら、起動時の CLI / 環境変数 / 標準出力ログ
- [ ] `arecord -l` / `aplay -l` / `cat /proc/asound/cards`

---

## 6. 追加観点（任意）

- [ ] 10分連続再生で問題が出ない
- [ ] ネットワーク経路（USB gadget/LAN）で UI が安定してアクセスできる
