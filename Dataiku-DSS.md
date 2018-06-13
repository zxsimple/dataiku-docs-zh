# Dataiku DSS #
欢迎来到Dataiku Data Science Studio (DSS)的参考文档。

更多资源请参考[Dataiku Learn](https://www.dataiku.com/learn/)。

- 安装DSS
  + 安装需求
  + 安装DSS新实例
  + 升级DSS实例
    + 更新DSS license
    + 其他安装选项
    + Hadoop和Spark集成配置
    + R集成
    + 自定义DSS实例安装
    + 安装数据库驱动
    + Java运行时
    + Python集成
    + 安装DSS插件
    + LDAP认证配置
    + 使用代理
    + 迁移操作
- [DSS概念](docs/DSS-concepts.md#DSS概念)
    + [数据集](docs/DSS-concepts.md#数据集)
    + [Recipes](docs/DSS-concepts.md#recipes)
    + [构建数据集](docs/DSS-concepts.md#构建数据集)
    + [被管控外部数据集](docs/DSS-concepts.md#被管控外部数据集)
    + [分区](docs/DSS-concepts.md#分区)
- [机器学习](docs/Machine-learning.md#机器学习)
    - [预测 (监督学习)](docs/MachineLearning/Supervised.md)
    - [聚类 (非监督学习)](docs/MachineLearning/Unsupervised.md)
    - [特征处理](docs/MachineLearning/Feature-handling.md)
    - [机器学习引擎](docs/MachineLearning/Engines.md)
    - [评分引擎](docs/MachineLearning/Scoring-engines.md)
- 连接数据
    + 支持的链接
    + 上传数据
    + 服务器文件系统
    + HDFS
    + Amazon S3
    + Google Cloud Storage
    + Azure Blob Storage
    + FTP
    + SCP / SFTP (aka SSH)
    + HTTP
    + SQL databases
    + Cassandra
    + ElasticSearch
    + 管控目录
    + 目录文件数据集
    + HTTP (带缓存)
    + 数据集插件
    + 数据连接宏
    + 创建重定位管控数据集
    + 数据排序
- 探索数据
    + 抽样
    + 分析
- Schema，存储类型和meanings
    + 定义
    + 基本用法
    + 数据准备schema
    + 创建数据集schema
    + 操作recipes的schema
    + 可识别meanings列表
- 数据准备
    + Processors参考
    + 过滤标记行
    + 管理日期
    + Reshaping
    + 处理地理位置
    + 抽样
    + 执行引擎
- 数据可视化
    + 抽样和图表引擎
    + 标准图表类型
    + 地理图表 (Beta)
    + 调色板
- 流图
    + 可视语法
    + 重构数据集
    + 并发控制
- 可视化recipes
    + Sync: 数据集拷贝
    + Grouping: 数据聚合
    + Window: 分析方法
    + Distinct: 计算唯一行
    + Join: 数据集连接
    + Split: 拆分数据集
    + Top N: 获取前N行
    + Stack: 数据合并
    + Sample: 数据集抽样
    + Sort: 数据排序
    + Pivot: 数据透视
    + Download: 数据下载
- 可编程recipes
    + 通用编辑器布局
    + Python recipes
    + R recipes
    + SQL recipes
    + Hive recipes
    + Pig recipes
    + Impala recipes
    + Spark-Scala recipes
    + PySpark recipes
    + Spark / R recipes
    + SparkSQL recipes
    + Shell recipes
    + 代码变量扩展recipes
- Notebook
    + SQL notebook
    + Python notebook
    + 预定义notebook
- Webapps
    + DSS webapps介绍
    + 标准 web apps
    + Shiny web apps
    + Bokeh web apps
    + 创建第一个webapp
    + 发布webapps到dashboard
    + 样例
- 编码报表
    + R markdown报表
- Dashboards
    + Dashboards概念
    + 显示设置
    + Insights参考
- 分区
    + 基于文件的数据集分区
    + SQL数据集分区
    + 设置分区依赖
    + 唯一标识分区
    + 数据集分区recipes
    + Hive分区recipes
    + SQL分区recipes
    + 分区变量替换
    + 两种分区模型
- DSS与Hadoop
    + Hadoop集成配置
    + 连接到安全集群
    + HDFS连接配置
    + DSS与Hive
    + DSS与Impala
    + Hadoop多用户安全
    + 不同Hadoop发行版
    + Hive数据集
    + 使用多Hadoop文件系统
    + Teradata Hadoop Connector
- DSS与Spark
    + DSS中Spark用法
    + Spark集成配置
    + Spark配置
    + 不同数据集类型用法
    + Spark pipelines
    + 限制和注意事项
- DSS与Python
    + 安装Python包
    + 重用Python代码
    + 使用Matplotlib
    + 使用Bokeh
    + 使用Plot.ly
    + 使用Ggplot
- DSS与R
    + 安装R包
    + 重用R代码
    + 使用ggplot2
    + 使用Dygraphs
    + 使用googleVis
    + 使用ggvis
- 代码运行环境
    + 操作(Python)
    + 操作(R)
    + 基础包
    + 使用Conda
    + 自动化节点
    + 非管控代码环境
    + 插件代码环境
    + 自定义选项环境
    + 问题处理
- 协同
    + 版本控制
- 插件
    + 安装插件
    + 离线安装插件
    + 实现自定义插件
    + 插件开发指南
- 自动化场景，指标和检查
    + 定义
    + 场景步骤
    + 启动场景
    + 场景运行报告
    + 自定义场景
    + 场景变量
    + 指标
    + 检查
    + 自定义检查
- 自动化节点和bundles
    + 安装自动化节点
    + 创建bundle
    + 导入bundle
- API节点: 实时服务
    + 介绍
    + 概念
    + 安装API节点
    + 创建第一个API service
    + 发布可视化预测模型
    + 发布Python预测模型
    + 发布R预测模型
    + 发布Python方法
    + 发布R方法
    + 发布SQL查询
    + 发布数据集lookup
    + 增强预测查询
    + API节点用户API
    + 使用apinode-admin工具
    + 高可用性和可伸缩性
    + endpoint管理版本
    + 日志和审计
    + 监控监控
- 高级主题
    + 抽样方法
    + Formula语言
    + 自定义变量扩展
- 文件格式
    + 分隔符文件(CSV / TSV)
    + 固定宽度
    + Parquet
    + Avaro
    + Hive SequenceFile
    + Hive RCFile
    + Hive ORCFile
    + XML
    + JSON
    + Excel
    + ESRI Shapefiles
- DSS内部API
    + Python API
    + Javascript API
    + R API
    + Scala API
- 公共API
    + 特性
    + 公共API Kyes
    + 公共API Python客户端
    + REST API
- 安全
    + 主权限
    + 连接安全
    + 用户档案
    + 发布对象
    + Dashboard授权
    + 多用户安全
    + 审计
    + 高级安全选项
    + 单点登录(SSO)
- DSS管理
    + dsscli tool
    + 数据目录
    + 备份
    + 登录DSS
    + DSS宏
    + 管理DSS磁盘利用率
- 问题解决
    + 诊断和调试
    + 获取支持
    + 通用问题
    + 错误编码
    + 已知问题
- Release notes
    + DSS 4.2 Release notes
    + DSS 4.1 Release notes
    + DSS 4.0 Release notes
    + DSS 3.1 Release notes
    + DSS 3.0 Relase notes
    + DSS 2.3 Relase notes
    + DSS 2.2 Relase notes
    + DSS 2.1 Relase notes
    + DSS 2.0 Relase notes
    + DSS 1.4 Relase notes
    + DSS 1.3 Relase notes
    + DSS 1.2 Relase notes
    + DSS 1.1 Release notes
    + DSS 1.0 Release Notes
    + Pre versions
- 其他文档
    + DSS早期版本
    + 其他Dataiku产品
- 第三方参考
    + Java库
    + Python库
    + Scala库
    + 本地库
    + R库
    + Javascript库
    + 数据
    + 仅限OS X版本
    + 定义
