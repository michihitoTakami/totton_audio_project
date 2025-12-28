# Docker（評価者向け / ソース不要）

このREADMEは **評価者（Trial）向け**です。**ソースコード不要**で、GHCR の image を pull して起動します。

---

## Jetson: runtime-only（image-based）

詳細な前提・ログ取得・既知トラブル・ロールバックは、まずこちら:

- [Jetson 評価者向け導入ガイド（ソース不要 / Docker）](../docs/jetson/evaluator_guide_docker.md)

最短コマンドだけ再掲:

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.runtime.yml up -d
docker compose -f jetson/docker-compose.jetson.runtime.yml logs -f
```

停止:

```bash
cd docker
docker compose -f jetson/docker-compose.jetson.runtime.yml down
```
