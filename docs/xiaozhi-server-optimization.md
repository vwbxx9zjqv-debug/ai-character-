# xiaozhi-esp32-server 低内存服务器优化记录

> 时间: 2026-07-06 | 服务器: 阿里云轻量 2C/4G

## 问题

运行 `xinnan-tech/xiaozhi-esp32-server` 时，服务器启动卡死，端口 8000/8003 无法监听，ESP32 设备无法连接。

### 根因

默认 ASR 模块 **FunASR** 加载 SenseVoiceSmall 全精度 `model.pt` (893MB)，Python 进程内存占用超 2.9GB。3.4GB 总内存不足，进程进入 `D` (不可中断 I/O) 状态卡死。

即使偶尔加载成功，推理 RTF 超过 1000（1 秒音频需 1000 秒处理），基本不可用。

```bash
$ ps -o pid,stat,rss,comm -p $(pgrep -f "python app.py")
# PID STAT   RSS COMMAND
# 3963 DLsl 2930020 python   ← D=不可中断等待, 2.9GB

$ ss -tlnp | grep -E "8000|8003"
# 无输出 — 端口未监听
```

## 解决方案

将 ASR 从 **FunASR** 切换为 **SherpaASR** (sherpa-onnx int8)。

### 修改配置

在 `data/.config.yaml` 中：

```yaml
selected_module:
  ASR: SherpaASR   # 原来是 FunASR（或不写，默认FunASR）
```

### sherpa-onnx 已内置

`core/providers/asr/sherpa_onnx_local.py` 已内置，首次启动自动从 ModelScope 下载 int8 模型 (~229MB)。

## 效果对比

| 指标 | FunASR (旧) | SherpaASR (新) |
|------|------------|----------------|
| 模型文件 | model.pt 893MB | model.int8.onnx 229MB |
| 精度 | FP32 | INT8 |
| 内存占用 | 2.9GB | ~860MB |
| 启动时间 | 卡死 | ~19秒 |
| 端口 | ❌ | ✅ 8000 + 8003 |

## 服务器复盘

```
FunASR → SherpaASR : 893MB → 229MB
内存 2.9G → 860MB (节省 70%)
启动 卡死 → 19秒
```

## 适用场景

- 阿里云/腾讯云/华为云轻量服务器 (≤4GB)
- 树莓派 4B (4GB)
- 任何内存 < 6GB 的环境

## 相关链接

- 上游仓库: https://github.com/xinnan-tech/xiaozhi-esp32-server
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
