## Totton Audio / web_brand

`web_brand/` は、Totton Audio の **1枚ペラのブランドページ**（静的HTML）です。

### 仕様

- **htmx + Alpine.js** を読み込み（軽量なインタラクション用途）
- **日本語 / 英語** の2言語
- **ブラウザ言語が日本語以外の場合は英語がデフォルト表示**
  - `?lang=ja|en` で上書き可能
  - 手動切替は `localStorage (totton.lang)` に保存
- お問い合わせリンクは Google Forms を使用
### GitHub Actions による同期

- `main` への push で `web_brand/**` に変更がある場合のみ、`michihitoTakami/totton-audio` リポジトリへ同期します。
- 秘密変数 `TOTTON_AUDIO_TOKEN`（`repo` 権限のPAT）が必要です。リポジトリ設定 → **Secrets and variables** → **Actions** で設定してください。
- 手動実行は GitHub の Actions ページから **Sync web_brand to totton-audio** を選び、`Run workflow` を押すだけです。
- 同期は `rsync --delete` 相当で不要ファイルを削除します。ターゲットリポジトリの独自ファイルは残らないため注意してください。

### ローカルでの確認方法

例えば以下で静的サーバーを立て、`web_brand/index.html` を開いてください。

```bash
cd web_brand
python3 -m http.server 8080
```
