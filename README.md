# TAAC-2021
2021腾讯广告算法大赛

## torch 版本 baseline
### 使用方法
1. 修改config/config.yaml中DatasetConfig的路径
2. 修改 main.py 中的 epoch_num或者其他配置
3. python main.py

### 完成进度
- 完成了video、audio 和 fusion 3个支路
- 完成了不同模块分别设置学习率，modal dropout，learning rate decay
- 完成了训练和验证两部分，可以计算gap

### 等待完成
- text 文本的支路，考虑采用bert和textcnn，fasttext等尝试一下
- 测试集的推理还未完成，还不能无法生成json文件
- 只支持单卡训练

### 模型性能
- 在使用video、audio和fusion的情况下，tf版本在验证集上能达到0.745，而torch的只能到0.729
- V100上训练100个epoch（14100个step）只需要30分钟，可能是因为还没有text

### 改进建议
1. 对比tf版本，看是否少了什么trick？
2. 参考石头哥的建议：学习率(warmup,restart)、扰动、层级mlp、asrocr用textCNN

