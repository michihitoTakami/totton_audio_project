# エラーコード体系設計書

> **Status:** Draft - レビュー中
> **Issue:** #208
> **Parent:** #207 (統一エラーハンドリング設計 EPIC)

## 1. 概要

本ドキュメントは、Magic Box Projectにおけるエラーコード体系の設計を定義する。
C++ Audio Engine から Web API クライアントまで、システム全体で一貫したエラーハンドリングを実現する。

### 1.1 設計目標

1. **一貫性**: 全レイヤーで同一のエラーコード体系を使用
2. **可読性**: 人間が読んでも、機械が解析しても理解しやすい
3. **拡張性**: 新しいエラーカテゴリ・コードを追加しやすい
4. **追跡性**: エラーの発生源を特定しやすい（inner_error による伝播）

### 1.2 準拠規格

- **RFC 9457**: Problem Details for HTTP APIs（旧 RFC 7807）
- **Google AIP-193**: エラーモデル設計ガイドライン
- **gRPC Status Codes**: ステータスコードの分類参考

## 2. エラーカテゴリ

エラーコードは5つの主要カテゴリと1つの予約カテゴリに分類される。各カテゴリは16ビットの上位4ビットで識別する。

| カテゴリ | プレフィックス | 数値範囲 | 説明 |
|---------|---------------|----------|------|
| Audio Processing | `AUDIO_` | 0x1000-0x1FFF | 入出力レート、フォーマット、バッファ関連 |
| DAC/ALSA | `DAC_` | 0x2000-0x2FFF | DACデバイス、ALSA関連 |
| IPC/ZeroMQ | `IPC_` | 0x3000-0x3FFF | プロセス間通信関連 |
| GPU/CUDA | `GPU_` | 0x4000-0x4FFF | CUDA、GPUメモリ関連 |
| Validation | `VALIDATION_` | 0x5000-0x5FFF | 入力検証、設定ファイル関連 |
| Internal | `INTERNAL_` | 0xF000-0xFFFF | 予約：未分類エラー、フォールバック用 |

### 2.1 カテゴリ選択ガイドライン

```
Q: どのカテゴリを使うべきか？

1. 入力サンプリングレートやフォーマットの問題 → AUDIO_
2. DAC/ALSAデバイスへのアクセス問題 → DAC_
3. ZeroMQやデーモンとの通信問題 → IPC_
4. CUDAの初期化やメモリ確保の問題 → GPU_
5. ユーザー入力や設定ファイルの問題 → VALIDATION_
```

### 2.2 Internal カテゴリについて

`INTERNAL` カテゴリは以下の用途で使用する**予約カテゴリ**である：

1. **未知のエラーコード**: C++側から受信したエラーコードがマッピングに存在しない場合
2. **Python側のみのエラー**: C++を経由しないPython固有のエラー
3. **フォールバック**: エラー伝播中にカテゴリを特定できない場合

**使用方針:**
- 新しいエラーが発生した場合、まず適切な主要カテゴリへの追加を検討する
- `INTERNAL` は一時的なフォールバックであり、恒久的な使用は避ける
- `INTERNAL` エラーが頻発する場合は、エラーコード体系の見直しを検討する

## 3. エラーコード一覧

### 3.1 Audio Processing (0x1000)

| コード | 名前 | HTTPステータス | 説明 |
|--------|------|----------------|------|
| 0x1001 | `AUDIO_INVALID_INPUT_RATE` | 400 | 入力サンプリングレートが無効（0, 負数, 非対応値） |
| 0x1002 | `AUDIO_INVALID_OUTPUT_RATE` | 400 | 出力サンプリングレートが無効 |
| 0x1003 | `AUDIO_UNSUPPORTED_FORMAT` | 400 | サポートされていないオーディオフォーマット |
| 0x1004 | `AUDIO_FILTER_NOT_FOUND` | 404 | 指定されたフィルタ係数ファイルが見つからない |
| 0x1005 | `AUDIO_BUFFER_OVERFLOW` | 500 | 内部バッファがオーバーフロー（処理が追いつかない） |
| 0x1006 | `AUDIO_XRUN_DETECTED` | 500 | ALSA XRUNが発生（アンダーラン/オーバーラン） |
| 0x1007 | `AUDIO_RTP_SOCKET_ERROR` | 500 | RTPソケット初期化またはマルチキャスト参加に失敗 |
| 0x1008 | `AUDIO_RTP_SESSION_NOT_FOUND` | 404 | 指定されたRTPセッションが存在しない |

### 3.2 DAC/ALSA (0x2000)

| コード | 名前 | HTTPステータス | 説明 |
|--------|------|----------------|------|
| 0x2001 | `DAC_DEVICE_NOT_FOUND` | 404 | 指定されたDACデバイスが見つからない |
| 0x2002 | `DAC_OPEN_FAILED` | 500 | DACデバイスのオープンに失敗 |
| 0x2003 | `DAC_CAPABILITY_SCAN_FAILED` | 500 | DACケイパビリティの取得に失敗 |
| 0x2004 | `DAC_RATE_NOT_SUPPORTED` | 422 | 指定されたサンプリングレートをDACがサポートしていない |
| 0x2005 | `DAC_FORMAT_NOT_SUPPORTED` | 422 | 指定されたフォーマットをDACがサポートしていない |
| 0x2006 | `DAC_BUSY` | 409 | DACデバイスが他のプロセスで使用中 |

### 3.3 IPC/ZeroMQ (0x3000)

| コード | 名前 | HTTPステータス | 説明 |
|--------|------|----------------|------|
| 0x3001 | `IPC_CONNECTION_FAILED` | 503 | ZeroMQソケットの接続に失敗 |
| 0x3002 | `IPC_TIMEOUT` | 504 | デーモンからの応答がタイムアウト |
| 0x3003 | `IPC_INVALID_COMMAND` | 400 | 不明なコマンドが送信された |
| 0x3004 | `IPC_INVALID_PARAMS` | 400 | コマンドのパラメータが不正 |
| 0x3005 | `IPC_DAEMON_NOT_RUNNING` | 503 | Audio Daemonが起動していない |
| 0x3006 | `IPC_PROTOCOL_ERROR` | 500 | ZeroMQプロトコルエラー（予期しないメッセージ形式） |

### 3.4 GPU/CUDA (0x4000)

| コード | 名前 | HTTPステータス | 説明 |
|--------|------|----------------|------|
| 0x4001 | `GPU_INIT_FAILED` | 500 | CUDAランタイムの初期化に失敗 |
| 0x4002 | `GPU_DEVICE_NOT_FOUND` | 500 | CUDAデバイスが見つからない |
| 0x4003 | `GPU_MEMORY_ERROR` | 500 | GPUメモリの確保に失敗 |
| 0x4004 | `GPU_KERNEL_LAUNCH_FAILED` | 500 | CUDAカーネルの起動に失敗 |
| 0x4005 | `GPU_FILTER_LOAD_FAILED` | 500 | フィルタ係数のGPUへのロードに失敗 |
| 0x4006 | `GPU_CUFFT_ERROR` | 500 | cuFFTライブラリでエラー発生 |

### 3.5 Validation (0x5000)

| コード | 名前 | HTTPステータス | 説明 |
|--------|------|----------------|------|
| 0x5001 | `VALIDATION_INVALID_CONFIG` | 400 | 設定ファイル（config.json）の形式が不正 |
| 0x5002 | `VALIDATION_INVALID_PROFILE` | 400 | EQプロファイルの形式が不正 |
| 0x5003 | `VALIDATION_PATH_TRAVERSAL` | 400 | パストラバーサル攻撃の検出（`..` を含むパス） |
| 0x5004 | `VALIDATION_FILE_NOT_FOUND` | 404 | 指定されたファイルが見つからない |
| 0x5005 | `VALIDATION_PROFILE_EXISTS` | 409 | 同名のプロファイルが既に存在（上書き不可） |
| 0x5006 | `VALIDATION_INVALID_HEADPHONE` | 404 | 指定されたヘッドホンがOPRA DBに存在しない |

## 4. エラー伝播フロー

### 4.1 レイヤー間のエラー伝播

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Web Browser                                    │
│  HTTP Response: application/problem+json                                │
│  {                                                                       │
│    "type": "/errors/dac-rate-not-supported",                            │
│    "title": "DAC Rate Not Supported",                                   │
│    "status": 422,                                                        │
│    "detail": "Sample rate 1000000 is not supported by DAC",             │
│    "error_code": "DAC_RATE_NOT_SUPPORTED",                              │
│    "category": "dac_alsa",                                              │
│    "inner_error": { "cpp_code": "0x2004", "alsa_errno": -22 }           │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ map_daemon_error()
┌─────────────────────────────────────────────────────────────────────────┐
│                        Python/FastAPI                                    │
│  - C++エラーコードをHTTPステータス + ProblemDetailに変換                │
│  - inner_error に下位レイヤーの詳細を保持                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ ZeroMQ JSON Response
┌─────────────────────────────────────────────────────────────────────────┐
│                          ZeroMQ IPC                                      │
│  {                                                                       │
│    "status": "error",                                                    │
│    "error_code": "DAC_RATE_NOT_SUPPORTED",                              │
│    "message": "Sample rate 1000000 is not supported",                   │
│    "inner_error": { "alsa_errno": -22, "alsa_func": "snd_pcm_hw_..." }  │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ ErrorCode enum + inner_error
┌─────────────────────────────────────────────────────────────────────────┐
│                        C++ Audio Engine                                  │
│  - ErrorCode::DAC_RATE_NOT_SUPPORTED (0x2004)                           │
│  - ALSA errno を inner_error として付加                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 inner_error の構造

下位レイヤーからのエラー情報を保持するための構造:

```json
{
  "inner_error": {
    "cpp_code": "0x2004",
    "cpp_message": "Rate 1000000 not in supported range",
    "alsa_errno": -22,
    "alsa_func": "snd_pcm_hw_params_set_rate_near",
    "cuda_error": null
  }
}
```

**inner_error フィールド:**

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `cpp_code` | string | C++エラーコード（16進数） |
| `cpp_message` | string | C++側のエラーメッセージ |
| `alsa_errno` | int \| null | ALSAエラー番号 |
| `alsa_func` | string \| null | 失敗したALSA関数名 |
| `cuda_error` | string \| null | CUDAエラー名（cudaSuccess等） |

## 5. HTTP ステータスコードマッピング

### 5.1 マッピング原則

| HTTPステータス | 用途 | エラーカテゴリ例 |
|----------------|------|-----------------|
| 400 Bad Request | クライアントの入力が不正 | VALIDATION_*, IPC_INVALID_* |
| 404 Not Found | リソースが存在しない | *_NOT_FOUND |
| 409 Conflict | リソース競合 | DAC_BUSY, VALIDATION_PROFILE_EXISTS |
| 422 Unprocessable Entity | 意味的に処理不可 | DAC_RATE_NOT_SUPPORTED |
| 500 Internal Server Error | サーバー内部エラー | GPU_*, AUDIO_BUFFER_OVERFLOW |
| 503 Service Unavailable | サービス利用不可 | IPC_DAEMON_NOT_RUNNING |
| 504 Gateway Timeout | タイムアウト | IPC_TIMEOUT |

### 5.2 リトライ可能性

| HTTPステータス | リトライ可能 | 推奨アクション |
|----------------|-------------|---------------|
| 400, 404, 409, 422 | No | ユーザーに修正を促す |
| 500 | Maybe | ログ確認後、サポートに連絡 |
| 503 | Yes | デーモン起動を待って再試行 |
| 504 | Yes | 時間をおいて再試行 |

## 6. 実装ガイドライン

### 6.1 C++ 実装 (`include/error_codes.h`)

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <system_error>

namespace AudioEngine {

enum class ErrorCode : uint32_t {
    OK = 0,

    // Audio Processing (0x1000)
    AUDIO_INVALID_INPUT_RATE      = 0x1001,
    AUDIO_INVALID_OUTPUT_RATE     = 0x1002,
    AUDIO_UNSUPPORTED_FORMAT      = 0x1003,
    AUDIO_FILTER_NOT_FOUND        = 0x1004,
    AUDIO_BUFFER_OVERFLOW         = 0x1005,
    AUDIO_XRUN_DETECTED           = 0x1006,

    // DAC/ALSA (0x2000)
    DAC_DEVICE_NOT_FOUND          = 0x2001,
    DAC_OPEN_FAILED               = 0x2002,
    DAC_CAPABILITY_SCAN_FAILED    = 0x2003,
    DAC_RATE_NOT_SUPPORTED        = 0x2004,
    DAC_FORMAT_NOT_SUPPORTED      = 0x2005,
    DAC_BUSY                      = 0x2006,

    // IPC/ZeroMQ (0x3000)
    IPC_CONNECTION_FAILED         = 0x3001,
    IPC_TIMEOUT                   = 0x3002,
    IPC_INVALID_COMMAND           = 0x3003,
    IPC_INVALID_PARAMS            = 0x3004,
    IPC_DAEMON_NOT_RUNNING        = 0x3005,
    IPC_PROTOCOL_ERROR            = 0x3006,

    // GPU/CUDA (0x4000)
    GPU_INIT_FAILED               = 0x4001,
    GPU_DEVICE_NOT_FOUND          = 0x4002,
    GPU_MEMORY_ERROR              = 0x4003,
    GPU_KERNEL_LAUNCH_FAILED      = 0x4004,
    GPU_FILTER_LOAD_FAILED        = 0x4005,
    GPU_CUFFT_ERROR               = 0x4006,

    // Validation (0x5000)
    VALIDATION_INVALID_CONFIG     = 0x5001,
    VALIDATION_INVALID_PROFILE    = 0x5002,
    VALIDATION_PATH_TRAVERSAL     = 0x5003,
    VALIDATION_FILE_NOT_FOUND     = 0x5004,
    VALIDATION_PROFILE_EXISTS     = 0x5005,
    VALIDATION_INVALID_HEADPHONE  = 0x5006,
};

// エラーコードを文字列に変換
// 未知のコードの場合は "UNKNOWN_ERROR" を返す
const char* errorCodeToString(ErrorCode code);

// カテゴリ名を取得
// 未知のコードの場合は "internal" を返す
const char* getErrorCategory(ErrorCode code);

// HTTPステータスコードに変換
// 未知のコードの場合は 500 を返す（Python側と同様のフォールバック）
int toHttpStatus(ErrorCode code);

// カテゴリ判定ヘルパー
constexpr bool isAudioError(ErrorCode code) {
    return (static_cast<uint32_t>(code) & 0xF000) == 0x1000;
}
constexpr bool isDacError(ErrorCode code) {
    return (static_cast<uint32_t>(code) & 0xF000) == 0x2000;
}
constexpr bool isIpcError(ErrorCode code) {
    return (static_cast<uint32_t>(code) & 0xF000) == 0x3000;
}
constexpr bool isGpuError(ErrorCode code) {
    return (static_cast<uint32_t>(code) & 0xF000) == 0x4000;
}
constexpr bool isValidationError(ErrorCode code) {
    return (static_cast<uint32_t>(code) & 0xF000) == 0x5000;
}

} // namespace AudioEngine
```

### 6.2 Python 実装 (`web/error_codes.py`)

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class ErrorCategory(str, Enum):
    """エラーカテゴリ"""
    AUDIO_PROCESSING = "audio_processing"
    DAC_ALSA = "dac_alsa"
    IPC_ZEROMQ = "ipc_zeromq"
    GPU_CUDA = "gpu_cuda"
    VALIDATION = "validation"
    INTERNAL = "internal"

class ErrorCode(str, Enum):
    """エラーコード（C++と同期）"""
    # Audio Processing
    AUDIO_INVALID_INPUT_RATE = "AUDIO_INVALID_INPUT_RATE"
    AUDIO_INVALID_OUTPUT_RATE = "AUDIO_INVALID_OUTPUT_RATE"
    AUDIO_UNSUPPORTED_FORMAT = "AUDIO_UNSUPPORTED_FORMAT"
    AUDIO_FILTER_NOT_FOUND = "AUDIO_FILTER_NOT_FOUND"
    AUDIO_BUFFER_OVERFLOW = "AUDIO_BUFFER_OVERFLOW"
    AUDIO_XRUN_DETECTED = "AUDIO_XRUN_DETECTED"

    # DAC/ALSA
    DAC_DEVICE_NOT_FOUND = "DAC_DEVICE_NOT_FOUND"
    DAC_OPEN_FAILED = "DAC_OPEN_FAILED"
    DAC_CAPABILITY_SCAN_FAILED = "DAC_CAPABILITY_SCAN_FAILED"
    DAC_RATE_NOT_SUPPORTED = "DAC_RATE_NOT_SUPPORTED"
    DAC_FORMAT_NOT_SUPPORTED = "DAC_FORMAT_NOT_SUPPORTED"
    DAC_BUSY = "DAC_BUSY"

    # IPC/ZeroMQ
    IPC_CONNECTION_FAILED = "IPC_CONNECTION_FAILED"
    IPC_TIMEOUT = "IPC_TIMEOUT"
    IPC_INVALID_COMMAND = "IPC_INVALID_COMMAND"
    IPC_INVALID_PARAMS = "IPC_INVALID_PARAMS"
    IPC_DAEMON_NOT_RUNNING = "IPC_DAEMON_NOT_RUNNING"
    IPC_PROTOCOL_ERROR = "IPC_PROTOCOL_ERROR"

    # GPU/CUDA
    GPU_INIT_FAILED = "GPU_INIT_FAILED"
    GPU_DEVICE_NOT_FOUND = "GPU_DEVICE_NOT_FOUND"
    GPU_MEMORY_ERROR = "GPU_MEMORY_ERROR"
    GPU_KERNEL_LAUNCH_FAILED = "GPU_KERNEL_LAUNCH_FAILED"
    GPU_FILTER_LOAD_FAILED = "GPU_FILTER_LOAD_FAILED"
    GPU_CUFFT_ERROR = "GPU_CUFFT_ERROR"

    # Validation
    VALIDATION_INVALID_CONFIG = "VALIDATION_INVALID_CONFIG"
    VALIDATION_INVALID_PROFILE = "VALIDATION_INVALID_PROFILE"
    VALIDATION_PATH_TRAVERSAL = "VALIDATION_PATH_TRAVERSAL"
    VALIDATION_FILE_NOT_FOUND = "VALIDATION_FILE_NOT_FOUND"
    VALIDATION_PROFILE_EXISTS = "VALIDATION_PROFILE_EXISTS"
    VALIDATION_INVALID_HEADPHONE = "VALIDATION_INVALID_HEADPHONE"

@dataclass
class ErrorMapping:
    """エラーコードとHTTPステータスのマッピング"""
    http_status: int
    category: ErrorCategory
    title: str

# エラーコード → HTTPステータス マッピング（全30コード）
ERROR_MAPPINGS: dict[ErrorCode, ErrorMapping] = {
    # Audio Processing (6コード)
    ErrorCode.AUDIO_INVALID_INPUT_RATE: ErrorMapping(400, ErrorCategory.AUDIO_PROCESSING, "Invalid Input Sample Rate"),
    ErrorCode.AUDIO_INVALID_OUTPUT_RATE: ErrorMapping(400, ErrorCategory.AUDIO_PROCESSING, "Invalid Output Sample Rate"),
    ErrorCode.AUDIO_UNSUPPORTED_FORMAT: ErrorMapping(400, ErrorCategory.AUDIO_PROCESSING, "Unsupported Audio Format"),
    ErrorCode.AUDIO_FILTER_NOT_FOUND: ErrorMapping(404, ErrorCategory.AUDIO_PROCESSING, "Filter Not Found"),
    ErrorCode.AUDIO_BUFFER_OVERFLOW: ErrorMapping(500, ErrorCategory.AUDIO_PROCESSING, "Buffer Overflow"),
    ErrorCode.AUDIO_XRUN_DETECTED: ErrorMapping(500, ErrorCategory.AUDIO_PROCESSING, "Audio XRUN Detected"),

    # DAC/ALSA (6コード)
    ErrorCode.DAC_DEVICE_NOT_FOUND: ErrorMapping(404, ErrorCategory.DAC_ALSA, "DAC Device Not Found"),
    ErrorCode.DAC_OPEN_FAILED: ErrorMapping(500, ErrorCategory.DAC_ALSA, "DAC Open Failed"),
    ErrorCode.DAC_CAPABILITY_SCAN_FAILED: ErrorMapping(500, ErrorCategory.DAC_ALSA, "DAC Capability Scan Failed"),
    ErrorCode.DAC_RATE_NOT_SUPPORTED: ErrorMapping(422, ErrorCategory.DAC_ALSA, "DAC Rate Not Supported"),
    ErrorCode.DAC_FORMAT_NOT_SUPPORTED: ErrorMapping(422, ErrorCategory.DAC_ALSA, "DAC Format Not Supported"),
    ErrorCode.DAC_BUSY: ErrorMapping(409, ErrorCategory.DAC_ALSA, "DAC Device Busy"),

    # IPC/ZeroMQ (6コード)
    ErrorCode.IPC_CONNECTION_FAILED: ErrorMapping(503, ErrorCategory.IPC_ZEROMQ, "Daemon Connection Failed"),
    ErrorCode.IPC_TIMEOUT: ErrorMapping(504, ErrorCategory.IPC_ZEROMQ, "Daemon Timeout"),
    ErrorCode.IPC_INVALID_COMMAND: ErrorMapping(400, ErrorCategory.IPC_ZEROMQ, "Invalid Command"),
    ErrorCode.IPC_INVALID_PARAMS: ErrorMapping(400, ErrorCategory.IPC_ZEROMQ, "Invalid Parameters"),
    ErrorCode.IPC_DAEMON_NOT_RUNNING: ErrorMapping(503, ErrorCategory.IPC_ZEROMQ, "Daemon Not Running"),
    ErrorCode.IPC_PROTOCOL_ERROR: ErrorMapping(500, ErrorCategory.IPC_ZEROMQ, "Protocol Error"),

    # GPU/CUDA (6コード)
    ErrorCode.GPU_INIT_FAILED: ErrorMapping(500, ErrorCategory.GPU_CUDA, "GPU Initialization Failed"),
    ErrorCode.GPU_DEVICE_NOT_FOUND: ErrorMapping(500, ErrorCategory.GPU_CUDA, "GPU Device Not Found"),
    ErrorCode.GPU_MEMORY_ERROR: ErrorMapping(500, ErrorCategory.GPU_CUDA, "GPU Memory Error"),
    ErrorCode.GPU_KERNEL_LAUNCH_FAILED: ErrorMapping(500, ErrorCategory.GPU_CUDA, "GPU Kernel Launch Failed"),
    ErrorCode.GPU_FILTER_LOAD_FAILED: ErrorMapping(500, ErrorCategory.GPU_CUDA, "GPU Filter Load Failed"),
    ErrorCode.GPU_CUFFT_ERROR: ErrorMapping(500, ErrorCategory.GPU_CUDA, "cuFFT Error"),

    # Validation (6コード)
    ErrorCode.VALIDATION_INVALID_CONFIG: ErrorMapping(400, ErrorCategory.VALIDATION, "Invalid Configuration"),
    ErrorCode.VALIDATION_INVALID_PROFILE: ErrorMapping(400, ErrorCategory.VALIDATION, "Invalid EQ Profile"),
    ErrorCode.VALIDATION_PATH_TRAVERSAL: ErrorMapping(400, ErrorCategory.VALIDATION, "Path Traversal Detected"),
    ErrorCode.VALIDATION_FILE_NOT_FOUND: ErrorMapping(404, ErrorCategory.VALIDATION, "File Not Found"),
    ErrorCode.VALIDATION_PROFILE_EXISTS: ErrorMapping(409, ErrorCategory.VALIDATION, "Profile Already Exists"),
    ErrorCode.VALIDATION_INVALID_HEADPHONE: ErrorMapping(404, ErrorCategory.VALIDATION, "Headphone Not Found in OPRA DB"),
}

def get_error_mapping(error_code: str) -> ErrorMapping:
    """
    エラーコードからマッピングを取得する。
    未知のエラーコードの場合はデフォルト（500 Internal Error）を返す。
    """
    try:
        code = ErrorCode(error_code)
        return ERROR_MAPPINGS.get(code, ErrorMapping(500, ErrorCategory.INTERNAL, "Internal Error"))
    except ValueError:
        # 未知のエラーコード
        return ErrorMapping(500, ErrorCategory.INTERNAL, "Unknown Error")
```

## 7. 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2024-XX-XX | 0.1.0 | 初版作成（レビュー用ドラフト） |

## 8. 参考資料

- [RFC 9457: Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457.html)
- [Google AIP-193: Errors](https://google.aip.dev/193)
- [gRPC Status Codes](https://grpc.io/docs/guides/status-codes/)
- [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines)
