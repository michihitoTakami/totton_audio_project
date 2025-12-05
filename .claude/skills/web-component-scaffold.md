# Web Component Scaffold

htmx + Alpine.js + Jinja2コンポーネントの新規作成を自動化します。

## Description

このSkillは、プロジェクトの標準的なWebコンポーネント（htmx + Alpine.js + Jinja2）を新規作成し、既存パターンに従ったテンプレートを生成します。DRY原則を徹底し、コンポーネントの再利用を促進します。

## Trigger Words

- `create component`
- `new web component`
- `scaffold ui`
- `コンポーネント作成`
- `新しいUI`

## Requirements

- Jinja2テンプレートエンジン
- htmx + Alpine.js統合済み
- 既存コンポーネント: `web/templates/components/`
- 実行ディレクトリ: プロジェクトルート

## Parameters

### 必須パラメータ

- **component_name**: コンポーネント名（スネークケース）
  - 例: `dac_selector`, `eq_slider`, `status_card`
  - 使用可能文字: `a-z0-9_`

- **component_type**: コンポーネントタイプ
  - `button`: ボタン（btn_primary等）
  - `card`: カードパネル
  - `input`: 入力フィールド（テキスト、数値）
  - `dropdown`: ドロップダウン選択
  - `toggle`: トグルスイッチ
  - `slider`: スライダー入力

### オプションパラメータ

- **properties**: プロパティリスト（JSON形式）
  - 例: `{"devices": "list", "selected": "str", "onChange": "function"}`

- **api_endpoint**: 連携するAPIエンドポイント
  - 例: `/api/dac/devices`

## Execution Steps

```bash
# 1. コンポーネント情報収集
# ユーザーから component_name, component_type, properties を取得

# 2. テンプレート生成
# web/templates/components/{component_name}.html を作成
# - Jinja2マクロ形式
# - 既存パターン（btn_primary, card_panel等）を踏襲

# 3. Alpine.jsロジック追加
# x-data, x-model, x-on:click 等の reactive binding

# 4. 使用例生成
# web/templates/pages/_example_{component_name}.html

# 5. APIエンドポイント確認（必要なら）
# web/routers/ に新規ルーター作成提案

# 6. 構文チェック
# Jinja2テンプレート解析
```

## Expected Output

### 成功時:
```markdown
# コンポーネント作成完了: dac_selector

## 生成されたファイル

1. **web/templates/components/dac_selector.html**
   - Jinja2マクロ: `{% macro dac_selector(devices, selected, on_change) %}`
   - Alpine.js: `x-data="dacSelectorData()"`
   - Props: devices (list), selected (str), on_change (function)

2. **web/templates/pages/_example_dac_selector.html**
   - 使用例とサンプルコード
   - Alpine.js初期化コード

## 使用方法

### テンプレートでのインポート

\`\`\`jinja2
{% from 'components/dac_selector.html' import dac_selector %}

<div class="dac-selection">
    {{ dac_selector(
        devices=available_devices,
        selected=current_device_id,
        on_change="handleDacChange"
    ) }}
</div>
\`\`\`

### Alpine.jsロジック

\`\`\`javascript
// web/static/js/components/dac_selector.js
function dacSelectorData() {
    return {
        devices: [],
        selected: '',

        async init() {
            await this.fetchDevices();
        },

        async fetchDevices() {
            try {
                const response = await fetch('/api/dac/devices');
                this.devices = await response.json();
            } catch (error) {
                console.error('Failed to fetch devices:', error);
            }
        },

        handleChange(deviceId) {
            this.selected = deviceId;
            // 親コンポーネントに通知
            this.$dispatch('dac-changed', { deviceId });
        }
    };
}
\`\`\`

## 次のステップ

1. **APIエンドポイント追加（必要な場合）**:
   \`\`\`bash
   # web/routers/dac.py に以下を追加
   @router.get("/devices", response_model=List[DacDevice])
   async def get_dac_devices():
       return get_available_dacs()
   \`\`\`

2. **スタイル追加**:
   \`\`\`css
   /* web/static/style.css */
   .dac-selector {
       /* スタイル定義 */
   }
   \`\`\`

3. **ページに統合**:
   - `web/templates/pages/system.html` に組み込み
   - 必要なイベントハンドラーを追加
```

## Error Handling

このSkillはベストエフォート戦略を採用しています：

1. **コンポーネント名重複時**:
   - 既存ファイルを表示
   - 上書き確認
   - バックアップ作成（`.bak`拡張子）

2. **プロパティ不足時**:
   - デフォルト値で生成
   - TODO コメント追加
   - 後で編集可能な形式

3. **API連携エラー時**:
   - スタブエンドポイント生成
   - モックデータを使用
   - TODO コメントで指示

4. **既存パターン不明時**:
   - 最もシンプルな button パターンを使用
   - カスタマイズ箇所をコメントで明示

## Best Practices

### プロジェクト固有のパターン

このSkillは以下の既存パターンを踏襲します：

1. **Jinja2マクロ形式**:
   ```jinja2
   {% macro component_name(param1, param2='default') %}
   <div class="component-wrapper">
       <!-- component HTML -->
   </div>
   {% endmacro %}
   ```

2. **Alpine.jsデータ構造**:
   ```html
   <div x-data="componentNameData()">
       <!-- reactive bindings -->
   </div>
   ```

3. **既存コンポーネントの再利用**:
   - `btn_primary`, `card_panel`, `slider_input` を優先使用
   - 新規作成は既存で対応できない場合のみ

### DRY原則の徹底

**⚠️ 重要**: 新しいUIを作る際は、必ず既存コンポーネントをチェック

既存コンポーネント（`web/templates/components/`）:
- `btn_primary` - プライマリボタン
- `card_panel` - カードパネル
- `slider_input` - スライダー入力
- `opra_search` - OPRA検索UI（dashboard/eq共通）

**ベタ書き禁止**: 必ずマクロとしてコンポーネント化

### イベント設計

Alpine.js イベント命名規則:
- `{component}-{action}` 形式
- 例: `dac-changed`, `eq-applied`, `status-refreshed`

## Related Skills

- `api-doc-sync`: 新規APIエンドポイント追加時は併用
- `worktree-pr-workflow`: コンポーネント作成後のPR作成

## Implementation Notes

このSkillは以下の既存構造を活用します：

### テンプレート構造
```
web/templates/
├── base.html                 # ベーステンプレート
├── components/               # 共通コンポーネント
│   ├── buttons.html          # ボタンマクロ
│   ├── cards.html            # カードマクロ
│   ├── inputs.html           # 入力フィールドマクロ
│   └── opra_search.html      # OPRA検索（参考実装）
└── pages/                    # ページテンプレート
    ├── dashboard.html
    ├── eq_settings.html
    └── system.html
```

### Alpine.js統合
```
web/static/js/
├── alpine-components.js      # Alpine.jsコンポーネント定義
└── opra_component.js         # OPRA検索ロジック（参考）
```

## Automation Level

**半自動実行**: テンプレート生成は自動、APIエンドポイントとスタイルは手動追加が必要です。
