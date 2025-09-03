CM-TFD：基于通道掩码的时频解耦多变量时序预测
（Channel-Masked Time-Frequency Disentanglement in Multivariate Time Series Forecasting）

此代码是我们的论文CM-TFD：基于通道掩码的时频解耦多变量时序预测 的 PyTorch 实现。

如果您觉得此项目有帮助，请不要忘记给它一个⭐星标以表示您的支持。谢谢！

CM-TFD（Channel-Masked Time-Frequency Disentanglement in Multivariate Time Series Forecasting）采用主-辅协同双分支结构：主分支集成混合专家建模与时频级联解耦机制，聚焦关键时序特征提取从而提升对复杂时序依赖关系的建模能力；辅分支构建可学习通道掩码，从原始输入中筛除冗余通道，并调控主分支的建模路径以提升结构稀疏性与相关性。此外，在训练过程中，同时利用时频域分析来增强预测范式，并通过动态调节时频域损失权重，有效缓解标签自相关导致的监督偏差。

快速入门

重要！！！本项目在 Python 3.8 下进行了全面测试，建议将 Python 版本设置为 3.8。

1.要求

给定一个 python 环境（注意：此项目在 python 3.8 下进行了全面测试），使用以下命令安装依赖项：

pip install -r requirements.txt

2.数据准备

经过良好预处理的数据集已经放在XX目录下

3.训练和评估模型

您可以使用如下命令将所有实验结果重现./scripts/multivariate_forecast

或者进行某个数据集的实验结果的复现，例如sh ./scripts/multivariate_forecast/ETTh1_script/CMTFD.sh

结果

我们利用时间序列预测基准 （TFB） 代码存储库作为统一的评估框架，提供对所有基线代码、脚本和结果的访问。按照 TFB 中的设置，我们不会应用“Drop Last”技巧来确保公平比较。



