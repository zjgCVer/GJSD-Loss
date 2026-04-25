# GJSD Loss for Oriented Object Detection in Remote Sensing Images

> 基于 MMRotate 的 GJSD Loss 官方实现。当前仓库提供损失函数代码与训练配置；论文引用信息与预训练权重将在论文正式发布后更新。

## 简介

本项目实现了面向遥感旋转目标检测的 **Geometric Jensen--Shannon Divergence (GJSD) Loss**。该损失在精度矩阵域中引入归一化几何混合分布，得到多元高斯分布之间的闭式、对称且具备等比缩放不变性的回归度量，可作为现有旋转检测器中回归损失的直接替换项。

<div align="center">
  <img src="docs/fig1.svg" alt="GJSD motivation" width="85%">
</div>


## 主要特点

- 闭式高斯度量：避免标准 JSD 中算术混合导致的非闭式问题。
- 对称比较：预测分布与目标分布在损失定义中处于对等位置。
- 等比缩放不变：几何关系不变时，整体尺度变化不会改变度量响应。
- 即插即用：仅替换回归损失，不改动主干、检测头、样本分配和推理流程。
- MMRotate 兼容：损失代码和训练配置均嵌入 MMRotate 工程结构。

## 项目结构

```text
GJSD-Loss/
├── configs/gjsd/
│   ├── roi-trans-r50_fpn_gjsd_1x_dota_le90_ms.py
│   └── s2anet_r50_fpn_gjsd_1x_dota_le135_ms.py
├── mmrotate/models/losses/gaussian_dist_loss_v1.py
├── docs/fig1.png
├── docs/fig3.png
└── README.md
```

## 安装

```bash
conda create -n gjsd python=3.10 -y
conda activate gjsd
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0rc6,<3.2.0"
pip install -v -e .
```

## 训练

```bash
python tools/train.py configs/gjsd/roi-trans-r50_fpn_gjsd_1x_dota_le90_ms.py
python tools/train.py configs/gjsd/s2anet_r50_fpn_gjsd_1x_dota_le135_ms.py
```

## 定性结果

<div align="center">
  <img src="docs/fig2.svg" alt="Qualitative results" width="95%">
</div>


## 说明

当前版本暂不包含预训练权重和论文 BibTeX。项目基于 MMRotate 构建，许可证请参考 `LICENSE`。
