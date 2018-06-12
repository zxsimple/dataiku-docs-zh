# 聚类（非监督学习）

聚类（又叫非监督学习）用于理解用户数据结构。例如，基于用户的支付历史信息可以将用户聚类成多个组，这对与营销策略很有用。

> 注意：不同于监督学习，非监督学习步需要**目标**变量。

- [在DSS中运行非监督学习](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html#running-unsupervised-machine-learning-in-dss)
- [抽样](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html#sampling)
- [特征处理](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html#features)
- [特征降维](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html#dimensionality-reduction)
- [异常检测](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html#outliers-detection)
- [算法](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html#algorithms)

## [在DSS中运行非监督学习](#id1)

按以下步骤在DSS中使用非监督学习：

- 进入项目中的Flow
- 点击要使用的数据集
- 选择*实验室*
- 创建可视化分析
- 点击模型Tab
- 选择*创建第一个模型*
- 选择*聚类*

## [抽样](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html#id2)

> 注意：在模型 > 设置 > 抽样下进行抽样设置

可用的的抽样方法取决于[机器学习引擎](Engines.md)

如果数据没有加载在内存中，那么你可能希望在抽取的样本上做聚类。可以从最开始抽样数据（最快的方式）或者从整个数据集上随机抽样。

## [特征处理](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html#id3)

参考 [特征处理](Feature-handling.md)。

## [特征降维](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html#id4)

> 注意：在模型 > 设置 > 特征降维下进行特征设置

特征降维通过将所有相关的变量分组合并成“主成分”来降低特征数量，得到的主成分与原始数据的差异尽可能大。

PCA对于聚类最大的好处在于降低算法运行时间，尤其在高纬特征的情况下。

可以选择启用、禁用和两种来进行比较。

## [异常检测](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html#id5)

> 注意：在模型 > 设置 > 异常检测下对异常检测参数进行设置

在进行聚类分析时，它通常会建议进行异常检测。如果不做异常检测它会产生不平衡的聚类，或者存在许多小簇而其中某个簇几乎包含所有的数据集。

DSS通过执行一次预聚类，聚出较大数量的聚类，将其中的小簇认为是异常，当：

- 簇的大小小于某个阈值（例如：10）
- 簇内总的样本小于某个比例阈值（例如：1%）

一旦检测出异常，通常可以：

- 删除：删除异常值。
- 聚类：对所有检测出的异常值创建一个聚类。

## [算法](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html#id6)

> 注意：在模型 > 设置 > 算法下进行算法设置

DSS支持多种聚类算法，可以选择多个聚类算法来比较哪个算法在数据集上有更好的表现。

可用的的算法取决于[机器学习引擎](Engines.md)。

