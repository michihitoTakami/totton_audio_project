# カーネルモジュールビルド

## 概要

JetPack 6.x (L4T 36.x) のデフォルトカーネルには、UAC2 (USB Audio Class 2.0) モジュールが含まれていません。Magic BoxでUSB Audio Gadgetを使用するには、カーネルモジュールのビルドが必要です。

---

## 必要なモジュール

| モジュール | 機能 | デフォルト |
|-----------|------|-----------|
| `libcomposite.ko` | USB Gadget ConfigFS | ✓ 含まれている |
| `usb_f_ecm.ko` | Ethernet Control Model | ✓ 含まれている |
| `u_audio.ko` | USB Audio共通 | ✗ 要ビルド |
| `usb_f_uac1.ko` | USB Audio Class 1.0 | ✗ 要ビルド |
| `usb_f_uac2.ko` | USB Audio Class 2.0 | ✗ 要ビルド |

---

## ビルド手順

### 前提条件

- Jetson Orin Nano (JetPack 6.x インストール済み)
- インターネット接続
- 十分なディスク容量 (~30GB)

### 1. ビルド環境準備

```bash
# 必要なパッケージのインストール
sudo apt update
sudo apt install -y build-essential bc libncurses5-dev libssl-dev \
    flex bison libelf-dev git wget

# 作業ディレクトリ作成
mkdir -p ~/kernel-build
cd ~/kernel-build
```

### 2. カーネルソース取得

#### 方法A: NVIDIA公式ソースから

```bash
# JetPack 6.0 (L4T 36.3) の場合
L4T_VERSION="36.3"
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/sources/public_sources.tbz2

# 展開
tar xf public_sources.tbz2
cd Linux_for_Tegra/source
tar xf kernel_src.tbz2
cd kernel/kernel-jammy-src
```

#### 方法B: JetsonHacks Kernel Builder使用

```bash
# jetsonhacks/jetson-orin-kernel-builder を使用
git clone https://github.com/jetsonhacks/jetson-orin-kernel-builder.git
cd jetson-orin-kernel-builder

# スクリプト実行（ソース取得含む）
./getKernelSources.sh
```

### 3. カーネル設定

```bash
cd ~/kernel-build/kernel/kernel-jammy-src

# 現在の設定をベースにする
zcat /proc/config.gz > .config

# または、デフォルト設定を使用
make ARCH=arm64 tegra_defconfig
```

### 4. USB Audio モジュール有効化

```bash
# menuconfig を使用
make ARCH=arm64 menuconfig
```

以下の項目を有効化（`<M>` = モジュール）:

```
Device Drivers --->
    USB support --->
        <M> USB Gadget Support --->
            <M> USB Gadget precomposed configurations --->
                <M> Audio Gadget
                    [ ] UAC 1.0 (Legacy)    # チェックしない（UAC2のみ使用）
            USB Gadget functions configurable through configfs --->
                <M> USB Audio Class function
```

または、設定ファイルを直接編集:

```bash
# .config に以下を追加/変更
cat >> .config << 'EOF'
CONFIG_USB_CONFIGFS_F_UAC1=m
CONFIG_USB_CONFIGFS_F_UAC2=m
CONFIG_USB_F_UAC1=m
CONFIG_USB_F_UAC2=m
CONFIG_USB_U_AUDIO=m
EOF

# 設定の整合性確認
make ARCH=arm64 olddefconfig
```

### 5. モジュールビルド

```bash
# モジュールのみビルド（カーネル全体は不要）
make ARCH=arm64 -j$(nproc) modules

# または特定のモジュールのみ
make ARCH=arm64 -j$(nproc) M=drivers/usb/gadget modules
```

### 6. モジュールインストール

```bash
# モジュールを適切な場所にコピー
KERNEL_VERSION=$(uname -r)
MODULE_DIR="/lib/modules/${KERNEL_VERSION}/kernel/drivers/usb/gadget/function"

sudo mkdir -p "${MODULE_DIR}"
sudo cp drivers/usb/gadget/function/u_audio.ko "${MODULE_DIR}/"
sudo cp drivers/usb/gadget/function/usb_f_uac1.ko "${MODULE_DIR}/"
sudo cp drivers/usb/gadget/function/usb_f_uac2.ko "${MODULE_DIR}/"

# 依存関係更新
sudo depmod -a

# モジュール自動ロード設定
echo "usb_f_uac2" | sudo tee /etc/modules-load.d/uac2.conf
```

### 7. 動作確認

```bash
# モジュールロード
sudo modprobe usb_f_uac2

# 確認
lsmod | grep uac2
# 出力例: usb_f_uac2  28672  0
```

---

## クロスコンパイル（PC上でビルド）

開発PCでビルドし、Jetsonにコピーする方法:

### 前提条件

```bash
# Ubuntu 22.04 PC
sudo apt install -y gcc-aarch64-linux-gnu
```

### クロスコンパイル

```bash
cd ~/kernel-build/kernel/kernel-jammy-src

export ARCH=arm64
export CROSS_COMPILE=aarch64-linux-gnu-

# ビルド
make tegra_defconfig
# (上記の設定変更を適用)
make -j$(nproc) modules
```

### Jetsonへ転送

```bash
# SCPでコピー
scp drivers/usb/gadget/function/*.ko jetson@192.168.55.1:/tmp/

# Jetson上でインストール
ssh jetson@192.168.55.1
sudo cp /tmp/*.ko /lib/modules/$(uname -r)/kernel/drivers/usb/gadget/function/
sudo depmod -a
```

---

## 事前ビルド済みモジュール

将来的には、事前ビルド済みモジュールを提供予定:

```bash
# インストールスクリプト（将来）
curl -sL https://releases.magicbox.audio/modules/install.sh | sudo bash
```

---

## トラブルシューティング

### "module format" エラー

```
insmod: ERROR: could not insert module: Invalid module format
```

**原因**: カーネルバージョン不一致

**対策**:
```bash
# 現在のカーネルバージョン確認
uname -r

# モジュールのビルド対象バージョン確認
modinfo /path/to/module.ko | grep vermagic

# 一致していない場合は再ビルド
```

### "Unknown symbol" エラー

```
insmod: ERROR: could not insert module: Unknown symbol in module
```

**原因**: 依存モジュール未ロード

**対策**:
```bash
# 依存関係確認
modinfo usb_f_uac2.ko | grep depends

# 依存モジュールを先にロード
sudo modprobe libcomposite
sudo modprobe u_audio
sudo modprobe usb_f_uac2
```

### ビルドエラー: "scripts/sign-file" not found

```bash
# SSL開発ライブラリインストール
sudo apt install libssl-dev

# 再ビルド
make ARCH=arm64 scripts
make ARCH=arm64 -j$(nproc) modules
```

---

## 参考リンク

- [NVIDIA Kernel Customization](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/Kernel/KernelCustomization.html)
- [JetsonHacks Kernel Builder](https://github.com/jetsonhacks/jetson-orin-kernel-builder)
- [Linux USB Gadget API](https://www.kernel.org/doc/html/latest/driver-api/usb/gadget.html)

---

## 関連ドキュメント

- [composite-gadget.md](./composite-gadget.md) - Composite Gadget設計
- [known-issues.md](./known-issues.md) - 既知の制限
