# Examples / 示例

- `polarity/event.json`: synthetic geometry and U probabilities; no waveform dependency. / 合成几何和 U 概率，不依赖波形文件。
- `synthetic/event.json`: runnable joint CAP + first-motion example with `observations.npz` and `truth.json`. / 可直接运行的联合例子，含波形与真值。
- `csndata_polarity/`: generated locally by the data-preparation script; intentionally ignored by Git. / 由数据准备脚本在本机生成，不提交真实波形。

Run from the repository root / 在仓库根目录运行：

```bash
python -m seismic_joint invert examples/polarity/event.json --mode polarity --device cuda:0 --output results/polarity
python -m seismic_joint invert examples/synthetic/event.json --mode joint --device cuda:0 --output results/joint
```

Input paths are relative to each event JSON. The provided synthetic first-motion probabilities come from the known mechanism and neural rays; they are not CNN outputs on the low-rate CAP records.

输入文件相对于事件 JSON 解析。合成示例的初动概率来自已知机制及神经网络射线，不是对低采样率 CAP 波形运行 CNN 的输出。
