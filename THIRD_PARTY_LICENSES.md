## THIRD_PARTY_LICENSES

本ファイルは、このプロジェクトで利用・同梱している第三者ソフトウェア／データのライセンス情報をまとめたものです。

---

## OPRA Project

- **URL**: `https://github.com/opra-project/OPRA`
- **ライセンス**:
  - **データ（manufacturer/product/EQ 等）**: CC BY-SA 4.0
  - **コード**: MIT
- **帰属表示**: `NOTICE.md` および OPRA の `README.md` に従ってください。

本プロジェクトでは、`OPRA sync` で取得したキャッシュ（`opra/versions/<commit>/database_v1.jsonl`）を利用します。取得元のコミット/URL/ハッシュは `metadata.json` に記録されます。開発・CI用には軽量フィクスチャ `tests/python/fixtures/opra/database_v1.sample.jsonl` を使用できます（`OPRA_DATABASE_PATH`）。

---

## HUTUBS (Head-related Transfer Function Database)

- **URL**: `https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960`
- **ライセンス**: CC BY 4.0
- **推奨文献**: Brinkmann et al., JAES 2019（詳細は `NOTICE.md`）

---

## 主要ライブラリ（ビルド/実行時）

注意: 下記は本リポジトリのビルド/実行に使われる依存関係の代表例です。配布形態（静的/動的リンク、同梱有無）により要求事項が変わり得るため、最終的な配布物に合わせて確認してください。

- **ALSA (alsa-lib)**: LGPL-2.1
- **ZeroMQ (libzmq)**: LGPL-3.0
- **libsndfile**: (システム提供パッケージのライセンスに従う)
- **NVIDIA CUDA / cuFFT**: NVIDIA EULA
- **Python packages**: `pyproject.toml` / `uv.lock` を参照

---

## Python: pysofaconventions

- **用途**: SOFA(HRTF) 読み込み
- **ライセンス**: BSD-3-Clause

---

## OPRA 内の第三者情報

OPRA データベースには第三者データ（例: AutoEQ）由来のデータが含まれます。詳細は OPRA リポジトリ同梱の `THIRD_PARTY_LICENSES.md` を参照し、必要に応じて追加の帰属表示を行ってください。
