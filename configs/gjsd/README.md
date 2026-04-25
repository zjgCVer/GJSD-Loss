# GJSD Loss Configs

This folder provides representative configs for training oriented detectors with GJSD Loss.

| Config | Detector | Dataset | Angle version |
| --- | --- | --- | --- |
| `roi-trans-r50_fpn_gjsd_1x_dota_le90_ms.py` | RoI Transformer | DOTA-v1.0 | le90 |
| `s2anet_r50_fpn_gjsd_1x_dota_le135_ms.py` | S$^2$A-Net | DOTA-v1.0 | le135 |

Example:

```bash
python tools/train.py configs/gjsd/roi-trans-r50_fpn_gjsd_1x_dota_le90_ms.py
```

The GJSD implementation is located at:

```text
mmrotate/models/losses/gaussian_dist_loss_v1.py
```
