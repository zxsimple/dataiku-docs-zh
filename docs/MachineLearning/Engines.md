# 机器学习训练引擎

DSS可视化机器学习引擎支持4种不同的机器学习引擎：

- [Scikit-learn / XGBoost](Engines/Sklearn.md)
- [MLLib (Spark)](Engines/MLlib.md)
- [H2O (Sparkling Water)](Engines/H2O.md)
- [Vertica](Engines/Vertica)

在创建模型的时候可以选择对应机器学习训练引擎来训练这个模型。

模型训练好后，就可以应用在新的记录上做预测。这个过程叫评价，通过各种[评价引擎](Scoring-engines.md)来评价。