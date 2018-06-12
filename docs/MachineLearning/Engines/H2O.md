#H2O (Sparkling Water)#

Sparkling Water是[H2O](http://h2o.ai/)基于Spark的机器学习引擎。

DSS通过在Spark集群上创建H2O集群训练H2O算法，与Spark无缝集成使得用户既可以用传统Spark MLLib，也可以用H2O提供的其他能力。

> 警告：分布式机器学习的开销是不能忽略的。
>
> 如果你的数据是存在内存中，就应考虑使用常规内存机器学习以便在更广泛的的选项和算法上更快地进行训练。

- [安装](#安装)

- [使用Sparkling Water](#使用Sparkling-Water)
- [分类算法](#分类算法)
- [聚类算法](#聚类算法)
- [限制](#限制)
- [内存需求](#内存需求)

## 安装

为了使用Sparkling Water首先得要有一个安装好的Spark集群，同时配置DSS集成Spark集群。了解更多关于DSS中Spark，请参考[DSS与Spark](https://doc.dataiku.com/dss/latest/spark/index.html)。

在DSS数据目录汇总运行以下脚本来安装Sparkling Waster：

```
./bin/dssadmin install-h2o-integration
```

如果不指定其他参数，DSS将会下载已安装Spark对应的Sparking Water版本JAR包。

如果你的机器没有联网，需要手动提供Sparkling Water的分发包，通过 `-sparklingWaterDir /path/to/sparkling-water` 选项指定Sparkling Water的解压路径。

## 使用Sparkling Water

通过创建分析，创建模型，选择H2O后端来训练H2O算法模型。

## 分类算法

H2O支持以下分类算法：

- 深度学习 (回归和分类)
- GBM (回归和分类)
- GLM (回归和分类)
- 随机森林 (回归和分类)
- 朴素贝叶斯 (多分类)

## 聚类算法

H2O支持如下聚类算法：

- KMeans (聚类)

## 限制

限制与[MLLib限制](https://doc.dataiku.com/dss/latest/machine_learning/mllib.html#mllib-limitations)相同。
除此之外，用户需要注意：

- 朴素贝叶斯算法只适用于分类的变量。
- 因为实现的缺陷，H2O的GLM算法不能很好处理未处理的分类变量。建议使用实体模型，它在算法性能效率上有相同的表现，但是没有错误的风险。
- H2O集群的UI(通常通过54321端口范围)将不能访问。

##内存需求##

不同于MLLib，Sparkling Water要求整个训练数据集都存放在分布式内存中(所有Saprk executor的内存之和)。

如果内存分配不足，Spark executor作业将会失败或者挂起。可能需要调优 ``spark.executor.memory`` 选项的值。