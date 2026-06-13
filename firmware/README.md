# ESP32-S3 固件 — Virtual Character Companion

基于 [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) (v2.2.6) 改造。

## 改动说明

替换了原项目的 MQTT 云服务协议，改为直连本地 WebSocket V3 后端。

| 文件 | 改动 |
|------|------|
| `main/protocols/ws_v3_protocol.h` | WebSocket V3 协议头文件（新增） |
| `main/protocols/ws_v3_protocol.cc` | WebSocket V3 协议实现（新增） |
| `main/application.cc` | 跳过云端激活，InitializeProtocol() 用 WsV3Protocol |
| `main/CMakeLists.txt` | 添加 ws_v3_protocol.cc |
| `sdkconfig.board` | 板级配置（ST7789 + PSRAM + 编译选项） |

## 安装方法

```bash
# 1. 克隆原版 xiaozhi-esp32
git clone https://github.com/78/xiaozhi-esp32.git
cd xiaozhi-esp32
git checkout b28bfe0  # 我们基于这个 commit 开发

# 2. 复制固件文件
cp -r /path/to/firmware/* .

# 3. 编译（需要 ESP-IDF v5.5.2）
IDF_MAINTAINER=1 idf.py build

# 4. 烧录
python -m esptool --chip esp32s3 --port /dev/cu.usbmodem* \
  --baud 921600 --before default_reset --after hard_reset \
  write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m \
  0x0 build/bootloader/bootloader.bin \
  0x8000 build/partition_table/partition-table.bin \
  0xd000 build/ota_data_initial.bin \
  0x20000 build/xiaozhi.bin \
  0x800000 build/generated_assets.bin
```

## 硬件

- ESP32-S3 N16R8 (16MB Flash, 8MB PSRAM)
- ST7789 LCD 240×320 (SPI3)
- INMP441 I2S 麦克风 + MAX98357 功放
