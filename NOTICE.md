## NOTICE / 帰属表示

Copyright (c) 2025 Michihito Takami

本プロジェクト（Totton Audio Project / GPU Upsampler）には、第三者が提供するソフトウェアおよびデータが含まれます。
第三者部分には、それぞれのライセンスが適用されます。
本プロジェクトの非商用条項は第三者ソフトウェア／データには適用されず、各ライセンス（例: OPRA CC BY-SA 4.0, HUTUBS CC BY 4.0, De-limiter MIT）が優先します。第三者ライセンスに従う限り、商用利用を含む利用が可能です（必要な帰属表示・継承条件を遵守してください）。

- 本プロジェクトの配布物（作者が著作権を保有する部分）のライセンス（英語）: `LICENSE`
- 本プロジェクトの配布物（作者が著作権を保有する部分）のライセンス（日本語）: `LICENSE.ja.md`
- 第三者ライセンス一覧: `THIRD_PARTY_LICENSES.md`

---

## OPRA (EQ データベース)

- **プロジェクト**: OPRA Project
- **URL**: `https://github.com/opra-project/OPRA`
- **データ（manufacturer/product/EQ 等）**: **CC BY-SA 4.0**
- **コード**: MIT

OPRA の帰属要件（例）:
- OPRA ロゴと説明、リポジトリへのリンク
- プリセット表示時は、OPRA だけでなく **プリセット作者** のクレジットを明示

※ OPRA の詳細な帰属要件は、OPRA リポジトリの `README.md` および sync 時に取得される `metadata.json`（source/commit/sha256）を参照してください。

---

## HUTUBS (HRTF データ)

- **データ**: HUTUBS - Head-related Transfer Function Database of TU Berlin
- **URL**: `https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960`
- **ライセンス**: **CC BY 4.0**
- **推奨文献**:
  - Brinkmann, F., Dinakaran, M., Pelzer, R., Wohlgemuth, J. J., Seipel, F., Voss, D., Grosche, P., & Weinzierl, S. (2019)
    “A Cross-validated database of measured and simulated HRTFs including 3D head meshes and anthropometric features.” Journal of the Audio Engineering Society.

---

## De-limiter モデル (jeonchangbin49/De-limiter)

- **プロジェクト**: `https://github.com/jeonchangbin49/De-limiter`
- **ライセンス**: **MIT**（`data/delimiter/weights/LICENSE.upstream` に同梱）
- **取得元**:
  - `all.pth` / `all.json`: `https://raw.githubusercontent.com/jeonchangbin49/De-limiter/main/weight/`
  - SHA256 (`tr -d ':'` で生ハッシュ): `df:d9:1f:96:05:c6:55:38:ac:ab:eb:9d:56:50:c2:11:19:61:1a:9e:04:93:de:3e:1f:30:05:76:a9:92:5e:c9` (all.pth), `b5:17:a0:19:1f:44:ed:88:9c:83:9d:79:df:f6:e0:07:5d:c5:40:57:d1:ab:0f:1d:55:58:1a:95:bb:90:ae:27` (all.json)
- **配置規約**: `data/delimiter/weights/<model_id>/<sr>/`（現行: `jeonchangbin49-de-limiter/44100/`）
- **配布方法**: git 非管理。`scripts/delimiter/download_assets.py` で取得し、`scripts/delimiter/export_onnx.py` で ONNX (`delimiter.onnx`) をローカル生成。

---

## 免責

本プロジェクトは「現状有姿（AS IS）」で提供されます。
利用によって生じたいかなる損害についても、作者は責任を負いません（詳細は `LICENSE` / `LICENSE.ja.md` を参照）。
