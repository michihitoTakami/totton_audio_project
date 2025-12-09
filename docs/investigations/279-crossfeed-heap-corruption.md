# Issue #279: クロスフィード有効化時のヒープ破壊

## 概要

Web UIからクロスフィード設定を有効化すると、デーモンがヒープ破壊でクラッシュする問題。

## エラーメッセージ

```
malloc(): mismatching next->prev_size (unsorted)
```

または

```
CUDA error at cudaHostRegister crossfeed stream output L: part or all of the requested memory range is already mapped
```

## 根本原因

### 原因1: 非同期転送の競合

`crossfeed_engine.cu` で `cudaMemcpyAsync` を使用した後、`cudaStreamSynchronize` を呼ばずに `std::vector::resize()` を実行していた。これにより:

1. GPU↔ホスト間のDMA転送が進行中
2. `resize()` によりベクタの内部バッファが再割り当て
3. DMAが解放済みメモリに書き込み
4. ヒープメタデータ破壊

### 原因2: ローカル変数のCUDAピンメモリ登録

`alsa_daemon.cpp` のコールバック内で `cf_output_left/right` がローカル変数として宣言されていた:

```cpp
// 問題のあるコード
std::vector<float> cf_output_left, cf_output_right;  // ローカル変数
```

これにより:
1. コールバック終了時にローカル変数が破棄
2. CUDAピンメモリ登録は解除されない
3. 次のコールバックで同じアドレスが再利用される可能性
4. `cudaHostRegister` が「already mapped」エラー

## 修正内容

### 修正1: cudaStreamSynchronize の追加

`src/gpu/crossfeed_engine.cu`:

```cpp
// 非同期転送後に同期を追加
checkCudaError(cudaStreamSynchronize(stream), "stream sync (crossfeed streaming)");
```

### 修正2: 出力バッファのグローバル化

`src/alsa_daemon.cpp`:

```cpp
// グローバル変数として宣言
static std::vector<float> g_upsampler_output_left;
static std::vector<float> g_upsampler_output_right;
static std::vector<float> g_cf_output_buffer_left;
static std::vector<float> g_cf_output_buffer_right;

// コールバック内で参照として使用
std::vector<float>& output_left = g_upsampler_output_left;
std::vector<float>& output_right = g_upsampler_output_right;
std::vector<float>& cf_output_left = g_cf_output_buffer_left;
std::vector<float>& cf_output_right = g_cf_output_buffer_right;
```

## テスト手順

### 1. ビルド

```bash
cd worktrees/279-crossfeed-heap-corruption
/usr/bin/cmake --build build -j8
```

### 2. デーモン起動

```bash
./build/gpu_upsampler_alsa > /tmp/alsa_daemon.log 2>&1 &
sleep 5
pgrep -f gpu_upsampler_alsa && echo "Running" || echo "Failed"
```


```bash
# リンク状態確認
pw-link -l

# 必要に応じて手動接続
pw-link "gpu_upsampler_sink:monitor_FL" "GPU Upsampler Input:input_FL"
pw-link "gpu_upsampler_sink:monitor_FR" "GPU Upsampler Input:input_FR"
```

### 4. クロスフィード有効化

```bash
uv run python3 -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect('ipc:///tmp/gpu_os.sock')
sock.send_json({'cmd': 'CROSSFEED_ENABLE'})
print('Response:', sock.recv_json())
sock.close()
ctx.term()
"
```

### 5. ストリーミングテスト

```bash
# デーモン状態確認
pgrep -f gpu_upsampler_alsa && echo "Daemon running" || echo "Daemon crashed!"

# ログ確認
cat /tmp/alsa_daemon.log | grep -E "state:|error|Error|CUDA"

# ストリーム状態が "streaming" になることを確認
```

### 6. クリーンアップ

```bash
pkill -f gpu_upsampler_alsa
rm -f /tmp/gpu_os.sock /tmp/alsa_daemon.log
```

## 期待される結果

- クロスフィード有効化: `{'message': 'Crossfeed enabled', 'status': 'ok'}`
- デーモン: クラッシュせず継続動作
- ログ: CUDAエラーなし

## 関連ファイル

- `src/gpu/crossfeed_engine.cu` - CUDAストリーム同期修正
- `src/alsa_daemon.cpp` - 出力バッファのグローバル化
- PR #283

## 教訓

1. **CUDAの非同期操作後は必ず同期**: `cudaMemcpyAsync` を使用する場合、ホストバッファを変更する前に `cudaStreamSynchronize` が必須
2. **CUDAピンメモリはライフサイクル管理が重要**: `cudaHostRegister` で登録したメモリは、解放前に必ず `cudaHostUnregister` が必要
3. **ローカル変数とCUDAの組み合わせは危険**: CUDAピンメモリを使用するバッファは永続化（グローバル/メンバ変数）すべき
