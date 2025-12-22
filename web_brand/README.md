## Totton Audio / web_brand

`web_brand/` は、Totton Audio の **1枚ペラのブランドページ**（静的HTML）です。

### 仕様

- **htmx + Alpine.js** を読み込み（軽量なインタラクション用途）
- **日本語 / 英語** の2言語
- **ブラウザ言語が日本語以外の場合は英語がデフォルト表示**
  - `?lang=ja|en` で上書き可能
  - 手動切替は `localStorage (totton.lang)` に保存
- お問い合わせリンクは Google Forms を使用

### ローカルでの確認方法

例えば以下で静的サーバーを立て、`web_brand/index.html` を開いてください。

```bash
cd web_brand
python3 -m http.server 8080
```
