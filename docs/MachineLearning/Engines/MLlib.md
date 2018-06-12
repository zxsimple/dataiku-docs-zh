# Spark MLLib #

[MLLib](http://spark.apache.org/mllib/)是Spark的机器学习库。DSS可以用它在大规模数据集上训练分类、聚类模型，而不需要把整个数据加载到内存中。

> 警告：分布式机器学习的开销是不能忽略的，而且它的支持也是有限制的(参考[限制](#限制))。
>
> 如果你的数据是存在内存中，就应考虑使用常规内存机器学习以便在更广泛的的选项和算法上更快地进行训练。

- [使用](#使用)
- [分类算法](#分类算法)
- [回归算法](#回归算法)
- [自定义模型](#自定义模型)
- [限制](#限制)

## 使用

当在分析中创建机器学习模型时，可以选择后段。默认的后段是*Python内存计算*引擎，如果你[配置了Spark](https://doc.dataiku.com/dss/latest/spark/installation.html#spark-setup)，就可以看到Spark MLLib选项。选择它之后模型就会在Spark上训练，建模的时候可以用MLLib可用模型或MLLib兼容的自定义模型。

你可以微调模型，在Flow中以重新训练模型发布，在用评分recipe在未标记的数据上进行预测。聚类模型也可以在新数据集上通过聚类recipe重新训练。

在模型训练、评分和聚类recipe设置上，有关于Spark的额外配置区域，你可以：

- 修改Spark基础配置
- 添加或者覆盖Spark配置选项
- 数据加载和就绪后选择存储级别来对数据集进行缓存
- 寻找Spark RDD分区数量用来对非HDFS输入数据集进行分区

关于DSS中Spark更多信息请参考[DSS与Spark](https://doc.dataiku.com/dss/latest/spark/index.html)。

## 分类算法

DSS 4.3支持Spark MLLib如下分类算法：

- 罗辑回归 (分类)
- 线性回归 (聚类)
- 决策树 (分类与回归)
- 随机森林 (分类与回归)
- 梯度提升输 (二分类与回归)
- 朴素贝叶斯 (多分类)
- 自定义模型

## 聚类算法

DSS 4.2中支持如下Spark MLLib聚类算法：

- KMeans (聚类)
- 高斯混合 (蕨类)自定义模型

## 自定义模型

用户通过自定义MLLib代码训练的模型，通过如下方法训练这种模型：

- 实现继承于`org.apache.spark.ml`包中的`Estimator`和`Model`类，这些类会用来训练你的模型：DSS会调用自定义`Estimator`的`fit(DataFrame)`方法和自定义`Model`的`transform(DataFrame)`方法。
- 将自定义类和所依赖类打到同一个jar包中并放在数据目录`lib/java`中。
- 在DSS中打开MLLib模型设置并添加一个自定义算法。
- 在代码编辑器中添加自定义`Estimator`的初始化(scala)代码，以及必要的`import`语句。初始化语句应该在最后面被调用。注意不建议在代码编辑器中使用申明类(包含匿名类)，因为它可能引起序列化错误。建议在jar文件中预先编译。

## 限制

由于通用[Spark的限制](https://doc.dataiku.com/dss/latest/spark/limitations.html)，MLLib有下面具体限制：

- MLLib中的梯度提升数不会输出每个分类的可能性，所以不能设置阈值，因此一些评估指标(AUC，Log loss，Lift)也是不可用的，同样的一些评估报告(变量重要性，决策图，提升图，ROC曲线)也是不可用的。
- 一些特征预处理选项也不可用的(可通过其他方法来实现)：
  - 特征合并
  - 除正则化之外的数值处理
  - 除虚拟编码之外的类型特征处理
  - 除分词、hash和统计之外的文本处理
  - 聚类降维
- 如果测试数据超过1百万条，基于性能和内存消耗考虑，将会抽样少于1百万的数据，因为一些评分方法需要排序和收集整个数据集。
- 不支持K-fold交叉验证和超参优化(grid search)。
