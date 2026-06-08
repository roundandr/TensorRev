# 后端扩展指南

本文档说明如何为 TensorRev 扩展一个新的 CUDA/PyTorch 兼容后端。当前 `mx` 后端是一个可参考的原型：它覆盖 MetaX/MACA C500 这类设备，通过 `TENSORREV_BACKEND=mx` 或自动识别进入 `hw/mx` 的扩展路径。

## 基本约定

- 后端 ID 使用小写短名，例如 `mx`。运行时通过 `TENSORREV_BACKEND=<backend>` 选择。
- `TENSORREV_BACKEND=auto` 是默认值。新后端需要在自动识别逻辑里根据设备名选择自己。
- 每个后端需要同时接入三层：Python 后端选择、qualifier/architecture 表、CUDA extension 构建和 dispatch。
- 新后端不应该破坏 NVIDIA 默认路径：`make`、`make f8`、`python run.py` 应继续按 NV 行为运行。

## 扩展 Checklist

1. 选择 backend ID
   - 在 `tensorrev/backend.py` 的 `SUPPORTED_BACKENDS` 中加入新 ID。
   - 为自动识别增加设备名关键词，例如当前 MX 使用 `MX_DEVICE_KEYWORDS = ("metax", "muxi", "mxc")`。
   - 在 `resolve_backend()` 中把对应设备名映射到新后端。

2. 添加 architecture 和 qualifier
   - 在 `tensorrev/common.py` 中新增 `<backend>_mma_qualifiers`。
   - 把新 qualifier 列表加入 `arch_mma_qualifiers`，使用清晰的 architecture key，例如当前 MX 使用 `"MX"`。
   - 在 `resolve_experiment_arch()` 中让新后端返回对应 architecture key。
   - 只列出后端已经实现的 qualifier。未实现的 dtype 或混合格式不要放进表里。

3. 添加 MMA dispatch
   - 在 `tensorrev/backend.py` 中为新后端添加 extension 加载逻辑。
   - 在 `get_mma_function(backend, dtype)` 中按 dtype 返回对应的 extension function。
   - 如果 extension 没有 build 或 JIT 失败，错误信息应明确指出需要构建哪个后端。

4. 添加 CUDA extension 源码
   - 新建 `hw/<backend>/gemm_setup.py`。
   - 将后端 kernel 放在 `hw/<backend>/` 下，例如当前 MX 使用 `hw/mx/mx_wmma_f16bf16tf32.cu` 和 `hw/mx/mx_mma_f8.cu`。
   - FP16/BF16/TF32 和 FP8 可以拆成不同 extension 模块，便于按 dtype dispatch。

5. 添加构建目标
   - 在顶层 `Makefile` 添加 `make <backend>` 入口。
   - 在 `hw/Makefile` 添加 `<backend>` target，并调用 `hw/<backend>/gemm_setup.py build_ext --inplace`。
   - 如需清理已有 `.so`，添加对应的 prepare/clean 逻辑。

6. 验证运行
   - 显式选择新后端：

     ```bash
     TENSORREV_BACKEND=<backend> python run.py
     ```

   - 如果支持自动识别，直接运行：

     ```bash
     python run.py
     ```

   - 输出中应能看到：

     ```text
     Selected TensorRev backend: <backend>
     ```

## CUDA Extension 合约

TensorRev 的 Python 层会把 MMA 调用统一成：

```python
D = fn(A, B.t().contiguous(), C)
```

因此每个后端的 extension function 应遵守以下约定：

- 接收三个 tensor：`A`、`B_col`、`C`。
- `B_col` 是 `B.t().contiguous()`，不是原始 `B`。
- 返回结果 tensor `D`。
- 输入 tensor 应在 CUDA 设备上，并满足 kernel 需要的 shape、dtype 和 contiguous 条件。
- 建议在 C++/CUDA extension 中使用 `TORCH_CHECK` 检查 dtype、shape、device 和 contiguous 状态。
- qualifier 字符串中的 dtype 会通过 `tensorrev/common.py` 中的 dtype metadata 转成 PyTorch dtype；dispatch 应与这些 dtype 保持一致。

## MX/MACA 示例

当前 MX/MACA 原型可以作为新后端模板：

- backend ID：`mx`
- 环境变量：`TENSORREV_BACKEND=mx`
- 设备自动识别关键词：`("metax", "muxi", "mxc")`
- architecture key：`"MX"`
- qualifier 表：`mx_mma_qualifiers`
- 源码目录：`hw/mx`
- 构建命令：`make mx`

当前 MX FP8 原型只支持 A/B 同类型 float8 输入，因此 qualifier 只包含：

- `m16n16k32.f32.e5m2.e5m2.f32`
- `m16n16k32.f32.e4m3.e4m3.f32`

不要在 qualifier 表中加入尚未实现的混合格式，例如 `e5m2/e4m3` 或 `e4m3/e5m2`。

## 最小验收标准

新增后端分支合入前，至少应确认：

- `TENSORREV_BACKEND=<backend> python run.py` 能选择新后端。
- `make <backend>` 能构建后端 extension，或运行时 JIT fallback 能成功编译。
- 新后端 qualifier 列表只包含已实现 kernel。
- NVIDIA 默认路径不回归：`make`、`make f8`、`python run.py` 仍保持原有行为。
- README 或相关文档说明了新后端的运行方式和已知限制。
