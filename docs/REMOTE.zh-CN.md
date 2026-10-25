# 已配置的远程 CUDA 主机

此页用于项目所有者现有的 `ssh cuda` 环境；其他机器请按仓库首页安装。

```bash
ssh cuda
cd /home/yuzy/focal_mechanism_joint_20261025
export SEISMIC_JOINT_PYTHON=/home/yuzy/.venvs/srl-cu128/bin/python
./run_cuda.sh arrivals --depth 10 --distance 30 100 300 --device cuda:0
./run_cuda.sh invert examples/synthetic/event.json \
  --mode joint --device cuda:1 --output results/demo_joint
```

该目录保留研究计算和真实 CSNDATA 示例。本 GitHub 仓库不提交真实波形：若需要重新准备该示例，且有 CSNDATA 与已有缓存访问权，可在原计算目录运行：

```bash
/home/yuzy/.venvs/srl-cu128/bin/python scripts/prepare_csn_polarity_example.py \
  --year 2022 --source /mnt/d/csndata \
  --cache /home/yuzy/csndata_polarity_cnn_20261024/cache
```

本页不包含 SSH 密码或密钥，也不替代个人账户/数据权限配置。
