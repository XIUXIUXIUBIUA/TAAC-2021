# TAAC-2021
- date: 2021-6-16
- author: mafp
- version: 2.0 正式版

## torch 版本 baseline
### 使用方法
1. init.sh 一键配置
2. setup.sh 安装所需的包（建议在notebook中运行）
3. 运行命令，理论上能得到0.789左右的模型

```bash
python -W ignore main.py   --device_ids 0 \ # 双卡则采用[0,1]
                            --pretrained_model ../checkpoint/50_wp.pt \ # 自监督预训练模型的路径
                            --saved_path ../checkpoint/0616/01/ # 模型保存路径
```

### 模型性能
- 单模型最高0.7894
- ensemble 可达0.795

### 改进建议
1. 自监督
2. 弱监督
3. 分类器改进
4. ensemble
5. 切分/多分辨率
6. 文本清洗、预处理
7. 关键帧提取、增加图像分支
8. 分析标签、数据
9. 视频裁剪再提取特征
### 迁移注意事项



