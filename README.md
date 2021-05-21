# TAAC-2021
- date: 2021-5-12
- author: mafp
- version: 1.0

## torch 版本 baseline
### 使用方法
1. 修改config/config.yaml中DatasetConfig的路径
2. 修改 main.py 中的 epoch_num或者其他配置
3. python main.py 
4. python inference.py

### 完成进度
- 完成了video audio text 和 fusion 3个支路
- 完成了不同模块分别设置学习率，modal dropout，learning rate decay，warmup
- 完成了训练、验证和测试整个流程
- 支持多卡训练（虽然速度并没有提升，建议还是使用单张V100）

### 等待完成
- 层级MLP、标签分层
- 扰动
- 日志

### 模型性能
- 在使用video和text两个分路时，在验证集上达到了0.83，但在测试集上只有0.766
- 训练1个epoch大约 1min15s

### 改进建议
1. 对比tf版本，看是否少了什么trick？
2. 参考石头哥的建议：学习率(warmup,restart)、扰动、层级mlp、asrocr用textCNN
3. 特征提取、视频裁剪

### 迁移注意事项
1. 注意需要在TTAC-2021的上级目录中存在 pretrained/bert，存放bert预训练模型，下载链接：https://huggingface.co/bert-base-chinese
2. pip install transformers
3. 注意路径修改



