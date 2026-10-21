# attention-kernel-lab

这是一个范围明确的注意力前向算子实验项目，重点是在线 softmax、GPU 分块、数值误差与可复现实验。包含可直接运行的 CPU 算法模型、PyTorch 参考实现、真实 Triton 内核、测试、基准脚本与 CI 配置。

[English README](README.md) · [设计说明](docs/design.md) · [复现步骤](docs/reproduce.md) · [限制](docs/limitations.md) · [实测记录](docs/validation.md)

**当前状态：** 初始电脑运行 Windows 10，显卡为 GeForce GT 730，驱动 388.13，不能验证本项目面向的 GPU 路径。GPU 编译、数值正确性和性能测试尚待合适设备执行。项目没有填写任何虚构的速度、TFLOPS、生产使用或公司内部成果。

## 已准备的内容

- 标准库实现的稠密 attention 与分块在线 softmax，用来检验算法和边界，不依赖 PyTorch。
- PyTorch 参考实现，以及显式的后端选择和输入约束检查。
- Triton 分块前向 attention，在线维护每行最大值、归一化因子与加权累加值。
- CPU 基础测试、可选的框架测试、仅在满足条件时运行的 GPU 测试。
- 先检查误差再计时的 GPU 基准脚本，保存环境、形状、误差与重复测量信息。
- 英文技术文档和 Linux/Windows CPU CI；GPU CI 需要用户自行准备专用 runner。

## 本机先运行 CPU 基础版本

在项目目录使用现有的 Python 3.10–3.12。若 `python` 只是 Windows 商店占位入口，请使用实际解释器路径；不需要为基础版本安装大型依赖。

```powershell
$env:PYTHONPATH = "src"
python -m attention_kernel_lab demo
python -m attention_kernel_lab doctor
python -m unittest discover -s tests -v
```

缺少 PyTorch 或合适 GPU 的测试会明确显示跳过。CPU 测试通过不能证明 GPU 内核已编译或已正确执行。实际执行结果以 [validation.md](docs/validation.md) 为准。

如需可安装的命令行入口，可在单独的虚拟环境执行 `python -m pip install -e .`。可选 CPU 框架测试使用 `requirements/cpu.txt`。GPU 环境请在另一台满足条件的 Linux CUDA 机器上按 [复现说明](docs/reproduce.md) 配置，本项目不会自动安装驱动、CUDA 或 WSL。

## GPU 接口的范围

| 项目 | 约束 |
| --- | --- |
| 输入 | `q[B,H,M,D]`、`k[B,H,N,D]`、`v[B,H,N,D]`，维度为正数 |
| 头维度 | `D` 为 16、32、64 或 128；batch、头数、头维度一致 |
| 存储和类型 | 同一设备上的连续四维张量；同一种 fp16 或 bf16 |
| 累加 | softmax 状态与输出累加使用 FP32，输出恢复为输入类型 |
| 因果掩码 | 左上对齐，只有 `j <= i` 的 key 可见；矩形 attention 也遵循此定义 |
| 非整块长度 | 掩码处理边界，不要求序列长度是 tile 的整数倍 |
| 启动边界 | `block_m` 和 `block_n` 为 16、32 或 64；`B*H <= 65535`；每个输入元素数小于 `2**31` |
| 平台目标 | Linux、NVIDIA CUDA、计算能力至少 8.0，且驱动与软件栈兼容 |
| 使用方式 | 仅前向推理；不包含反向、dropout、自定义 mask、GQA 或 KV cache |

计算能力检查只是准入条件，不表示所有较新架构都已支持。没有验证 Blackwell、AMD 或 MI300。显式指定 `backend="triton"` 时，不满足条件会报错；`auto` 可在不满足 GPU 条件时选择参考实现，但不会吞掉实际内核的编译或执行错误。

```python
from attention_kernel_lab.torch_ops import attention

# q/k/v 需要先按上述要求分配。
out = attention(q, k, v, causal=True, backend="triton")
```

## 如何理解后续性能结果

基准中的 `sdpa_auto` 指 PyTorch 自己选择后端的 SDPA，它可能已经采用融合内核。比较时必须同时记录显卡、驱动、PyTorch/Triton/CUDA 版本、输入形状、类型、因果模式和误差。JIT 与预热不计入稳定态计时；错误的结果不能进入性能对比。

更大的 tile 是否更快、寄存器压力是否抵消访存收益、因果模式能否减少无效工作，都需要实际测量。当前仓库没有性能结论。公开算法和资料来源见 [英文 README](README.md#sources-and-provenance)。代码不包含任何公司内部代码或虚构履历。

源码、配置和脱敏后的验证记录一并提供。GPU 编译、正确性与性能结果仍需要在满足条件的机器上实测。许可状态见 [NOTICE.md](NOTICE.md)，目前没有授予分发许可。
