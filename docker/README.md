# Docker 概要

- `jetson/` : Magic Box (Web + Daemon) コンテナ。RTP受信（GStreamer）は同一コンテナ内で実行する。
- `jetson_pcm_receiver/` : 旧TCPブリッジのDockerfile置き場（Composeからは除去済み、ビルド非推奨）
- `raspberry_pi/rtp_receiver/` : ラズパイ側で ZeroMQ ブリッジをサイドカーとして起動できる Python 実装（`python -m raspberry_pi.rtp_receiver`）。

ローカル検証用 (`docker/local`) と Raspberry Pi 用 (`docker/raspi`) は不要になったため削除しました。

## まずどれ？（入口）

用途ごとに README を分けています。

- **評価者（ソース不要 / 最短起動）**:
  - [Docker（評価者向け / ソース不要）](./README.evaluator.md)
- **開発者（ローカルビルド）**:
  - [Docker（ローカルビルド / 開発者向け）](./README.local_build.md)

## 補足: ディレクトリ説明

- `docker/jetson/` : Jetson 向け compose / Dockerfile / entrypoint
- `docker/jetson_pcm_receiver/` : 旧TCPブリッジのDockerfile置き場（Composeからは除去済み、ビルド非推奨）
