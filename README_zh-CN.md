<div align="center">
  <h1>GJSD Loss for Oriented Object Detection in Remote Sensing Images:<br>Symmetric and Scale-Invariant Gaussian Regression</h1>
  <h2>面向遥感图像旋转目标检测的 GJSD 损失：<br>对称且尺度不变的高斯回归</h2>
  <p>
    <b>基于 MMRotate 的 GJSD Loss 官方实现</b><br>
    当前仓库提供损失函数代码与训练配置；论文 BibTeX 引用将在论文正式发布后更新。
  </p>
</div>

---

## 简介

在遥感影像定向目标检测中，基于高斯建模的边界框回归将旋转边界框映射为二维高斯分布，从而能够在统一的概率框架下联合建模目标的位置、尺度与方向。然而，这类方法的有效性在很大程度上取决于用于比较预测分布与目标分布的度量方式。现有度量存在不同局限：Kullback--Leibler 散度具有方向依赖性，Gaussian Wasserstein 距离的响应受目标绝对尺度影响，而基于概率重叠的度量在大误差区域中可能提供较弱的监督信号。为此，本文提出一种用于旋转边界框回归的几何 Jensen--Shannon 散度损失，称为 GJSD Loss。通过引入在精度矩阵域中组合输入高斯分布的归一化几何混合，即通过其协方差矩阵的逆矩阵进行组合，GJSD 得到了多元高斯分布之间的闭式散度表达，并能够联合度量中心偏移、尺寸/形状变化和方向误差。所得损失具有对称性、非负性、诱导高斯层面的唯一零点以及尺度不变性，可直接作为现有定向检测器中的回归损失替换项使用。在受控替换协议下，多个检测器和多个数据集上的实验表明，GJSD Loss 能够取得具有竞争力的整体精度，并在高阈值定位指标上表现出更明显的增益。特别地，在 DOTA-v1.0 上，S<sup>2</sup>A-Net+GJSD 和 RoI Transformer+GJSD 分别取得 79.66%/58.47% 和 80.56%/56.23% 的 mAP/AP<sub>75</sub>；在 DIOR-R 上，RoI Transformer+GJSD 取得 66.02% mAP。

<div align="center">
  <h3>GJSD 的动机与构造</h3>
  <img src="docs/fig1.svg" alt="GJSD motivation and construction" width="85%">
</div>

---

## 主要特点

- **闭式高斯度量。** GJSD 以归一化几何混合替代标准 Jensen–Shannon Divergence 中的算术混合，从而避免高斯分布之间 JSD 通常不存在闭式解的问题。
- **对称比较机制。** 预测框分布与目标框分布在损失定义中处于对等位置，避免单向度量带来的方向偏置。
- **等比缩放不变性。** 当目标的相对几何关系保持不变时，整体坐标尺度变化不会改变原始 GJSD 度量及其有界训练损失响应。
- **即插即用。** 本仓库仅替换边界框回归损失，不改变检测器结构、训练流程、样本分配和推理阶段。
- **兼容 MMRotate。** GJSD 相关损失代码与训练配置已经嵌入 MMRotate 工程结构，便于复现实验和二次开发。

---

## 定性结果

<div align="center">
  <h3>DOTA-v1.0 上的检测可视化结果</h3>
  <img src="docs/fig2.svg" alt="Qualitative results on DOTA-v1.0" width="95%">
</div>

---

## 项目结构

```text
GJSD-Loss/
├── configs/
│   └── gjsd/
│       ├── roi-trans-r50_fpn_gjsd_1x_dota_le90_ms.py
│       └── s2anet_r50_fpn_gjsd_1x_dota_le135_ms.py
├── mmrotate/
│   └── models/
│       └── losses/
│           └── gaussian_dist_loss_v1.py
├── docs/
│   ├── fig1.svg
│   └── fig2.svg
├── tools/
│   ├── train.py
│   └── test.py
└── README.md
```

### 关键文件说明

| 文件 | 说明 |
| --- | --- |
| `mmrotate/models/losses/gaussian_dist_loss_v1.py` | 高斯距离类损失函数实现文件，包含 GJSD Loss 相关实现。 |
| `configs/gjsd/roi-trans-r50_fpn_gjsd_1x_dota_le90_ms.py` | RoI Transformer + GJSD 在 DOTA-v1.0 上的训练配置。 |
| `configs/gjsd/s2anet_r50_fpn_gjsd_1x_dota_le135_ms.py` | S<sup>2</sup>A-Net + GJSD 在 DOTA-v1.0 上的训练配置。 |
| `docs/fig1.svg` | GJSD 动机与方法构造示意图。 |
| `docs/fig2.svg` | DOTA-v1.0 定性检测结果图。 |

---

## 安装

本项目遵循 MMRotate 1.x 工程结构。推荐使用独立 Conda 环境进行安装。

```bash
conda create -n gjsd python=3.10 -y
conda activate gjsd

# 根据本机 CUDA 版本安装 PyTorch。
# 以下命令以 CUDA 11.8 为例：
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 安装 OpenMMLab 相关依赖。
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0rc6,<3.2.0"

# 安装本项目。
pip install -v -e .
```

开发实验环境使用 PyTorch 2.0.1、CUDA 11.8 和 NVIDIA RTX 3090。其他与 MMRotate 1.x 兼容的环境也可使用，但复现实验时建议尽量保持依赖版本一致。

---

## 数据准备

请按照 MMRotate 官方流程准备 DOTA-v1.0 数据集。默认单尺度数据目录建议采用以下结构：

```text
data/split_ss_dota/
├── train/
│   ├── images/
│   └── annfiles/
└── val/
    ├── images/
    └── annfiles/
```

若需要生成 DOTA 提交格式结果，请根据实际路径修改配置文件中的 `outfile_prefix`。

---

## 训练

### RoI Transformer + GJSD

```bash
python tools/train.py configs/gjsd/roi-trans-r50_fpn_gjsd_1x_dota_le90_ms.py
```

### S<sup>2</sup>A-Net + GJSD

```bash
python tools/train.py configs/gjsd/s2anet_r50_fpn_gjsd_1x_dota_le135_ms.py
```

---

## 测试

```bash
python tools/test.py configs/gjsd/roi-trans-r50_fpn_gjsd_1x_dota_le90_ms.py \
    work_dirs/roi-trans-r50_fpn_gjsd_1x_dota_le90_ms/latest.pth
```

如需测试 S<sup>2</sup>A-Net，请相应替换配置文件路径和权重文件路径。

---

## 预期结果

本项目强调受控损失替换实验。在相同检测器结构和训练设置下，仅将边界框回归损失替换为 GJSD Loss。

| 数据集 | 检测器 | 指标 | 结果 |
| --- | --- | --- | --- |
| DOTA-v1.0 | S<sup>2</sup>A-Net + GJSD | mAP | 79.66 |
| DOTA-v1.0 | S<sup>2</sup>A-Net + GJSD | AP<sub>75</sub> | 58.47 |
| DOTA-v1.0 | RoI Transformer + GJSD | mAP | 80.56 |
| DOTA-v1.0 | RoI Transformer + GJSD | AP<sub>75</sub> | 56.23 |
| DIOR-R | RoI Transformer + GJSD | mAP | 66.02 |

上述结果用于记录论文实验设置下的典型结果。由于硬件平台、软件版本、数据预处理流程和随机种子可能不同，复现实验结果可能存在小幅波动。

---

## 运行时说明

GJSD 在训练阶段的回归损失计算中引入 Cholesky 分解和线性方程求解操作。由于旋转框被表示为二维高斯分布，这些操作仅作用于较小的 2 × 2 正定矩阵，并且只针对正样本执行。

GJSD Loss 不引入额外可学习参数，也不增加推理阶段分支。因此，在推理阶段，其计算开销与原检测器保持一致。

---

## 仓库状态

- [x] GJSD Loss 实现
- [x] RoI Transformer + GJSD 配置
- [x] S<sup>2</sup>A-Net + GJSD 配置
- [x] 方法动机与定性结果图
- [ ] 论文 BibTeX 引用

当前版本暂不包含论文 BibTeX 引用。相关内容将在论文正式发布后更新。

---

## 致谢

本项目基于 [MMRotate](https://github.com/open-mmlab/mmrotate) 和 OpenMMLab 生态构建。感谢 MMRotate 团队提供的旋转目标检测代码库。

---

## 许可证

本仓库遵循原 MMRotate 项目的许可证。具体内容请参考 [LICENSE](LICENSE)。
