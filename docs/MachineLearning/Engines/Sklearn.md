# Scikit-learn / XGBoost #

大部分算法基于

绝大多数算法是基于 [Scikit Learn]() 或者 [XGBoost]() 机器学习库。

引擎提供了内存处理能力，训练集和测试集需要加载到内存中。如有需要可以使用抽样设置。

- 分类算法
  - (回归) 最小二乘法
  - (回归) 岭回归
  - (回归) Lasso回归
  - (分类) 罗辑回归
  - (分类与回归) 随机森林
  - (分类与回归) 梯度提升树
  - (分类与回归) XGBoost
  - (分类与回归) 决策树
  - (分类与回归) 支持向量机
  - (分类与回归) 随机梯度下降
  - (分类与回归) K近邻
  - (分类与回归) Extra Random Trees
  - (分类与回归) Artificial Neural Network
  - (分类与回归) Lasso Path
  - (分类与回归) 自定义模型
- 聚类算法
  - K-means
  - 混合高斯模型
  - Mini-batch K-means
  - 层次聚类(自底向上)
  - 谱聚类
  - DBSCAN
  - 交互式聚类 (两步聚类)
  - 孤立森林 (异常检测)
  - 自定义模型

## Prediction algorithms

这种引擎的分类算法支持以下算法。

### (回归) 最小二乘法

最小二乘法或者线性最小二乘法是最简单的线性回归算法。目标变量由输入变量加权求和得到。OLS通过cost函数计算合适的权重。

OLS非常简单而且是“可解释”的，但是：

- 它不能自动拟合目标变量，如果目标变量不是输入特征线性组合结果
- 它对输入数据集中的错误高度敏感且容易过拟合

参数：

- **并行度：**并行训练的核数。在训练大量数据集的情况下，用更多的核提高训练速度但会消耗更多内存。(-1代表所有核)

### (回归) 岭回归

岭回归通过对权重引入惩罚(正则化项)来解决最小二乘法的一些问题。岭回归使用L2正则化，L2正则化降低模型系数的大小。

参数：

- **正则化项 (自动优化或者确定值)：**自动优化一般比多个确定值速度要快，但是不支持稀疏特征(例如文本哈希) 
- **Alpha：**正则化项，可以指定以逗号分割的多个值列表。这会增加训练时间。

### (回归) Lasso回归


Lasso 回归是另外一种线性回归，它使用不同的正则化项(L1正则化) 。L1正则化降低最终模型的特征数量。

参数：

- **正则化项 (自动优化或者确定值)：**自动优化一般比多个确定值速度要快，但是不支持稀疏特征(例如文本哈希) 
- **Alpha：**正则化项，可以指定以逗号分割的多个值列表。这会增加训练时间。

### (分类) 罗辑回归

罗辑回归是采用线性模型的分类算法(通过将输入特征线性合并后计算目标特征)。罗辑回归最小化一个cost 函数(称为logit或者sigmoid函数)，使其更适用于分类。一个简单的逻辑回归算法容易过拟合同时对输入数据集中的错误也很敏感。为了解决这些问题，可以对权重使用惩罚(或者正则项)。

逻辑回归有两个参数：

- **正则化 (L1，L2 正则化)：**L1 正则化减少模型的特征数量。L2正则化降低每个特征的系数。
- **C：**错误项的惩罚参数C。较低的C值会产生一个平滑决策边界(较高方差)，较高的C值目的是将所有的训练数据都正确分类，但是有导致过拟合(较高偏差)。(C对应于正则化参数的逆)。可以用逗号隔开的列表指定多个C值。

### (分类与回归) 随机森林

决策树是一种只建立一棵决策树的分类算法。决策树的每个节点包含了某个输入特征中的条件。

回归决策树由多棵决策树组成。在预测过程中，每棵树独立预测，森林中的每棵树进行“投票”。森林对每棵树的结果进行平均。在森林成长(训练)过程中：

- 对于每棵树，随机抽样训练数据集；
- 在树的每个决策点，考虑随机选取输入特征中的一个子集。

参数：

- **树的数量：**森林中树的数量。增加随机森林中树的数量可以避免过拟合。可以用逗号隔开的列表指定多个值，这将会增加训练时长。
- **特征采样策略：**调整每次分裂的特征数量。
  - 自动选择30%的特征
  - 平方根或者以2为基数的对数
  - 特征数量的固定值
  - 特征数量的固定百分比
- **树的最大深度：**森林中每棵树的最大深度。较大的深度通常增加预测的准确度，但可能会导致过拟合。设置为0表示无限深度(树持续分裂指导每个节点只保留一个目标值)
- **每个叶子的最小样本数：**单个节点在分裂时所需要的最小样本数量。较小的值增加预测的准确度(通过分裂树)，但是会导致过拟合切增加训练和预测时间。
- **并行度：**并行训练的核数。在训练大量数据集的情况下，用更多的核提高训练速度但会消耗更多内存。(-1代表所有核)

## (分类与回归) 梯度提升树

梯度提升树是基于决策树的另外一种融合模型。通过一定顺序添加树，每棵树都会用于提升整个融合模型的性能。GBRT的优点在于：

- 天然处理混合数据类型(异构特征)
- 预测能力
- 对输出空间中异常值的鲁棒性(通过鲁棒的损失函数)

请注意有可能会面临扩展性问题，因为顺序提升的特效它很难做到并行。

梯度提升树有四个参数：

- **提升阶段数量：**要执行的提升阶段数量。梯度提升对过拟合鲁棒性很好，因此更大的值将会得到更好的结果。可以用逗号隔开的列表指定多个值，这将会增加训练时长。
- **学习率：**学习率通过learning_rate参数降低每棵树的贡献率。在学习率和提升阶段数量之间需要做个折中。更小的学习率需要更大的提升阶段数量。可以用逗号隔开的列表指定多个值，这将会增加训练时长。
- **Loss：**取决于是分类问题还是回归问题可供选择不同的损失函数。
  - **分类：**对数似然损失函数deviance(相当于逻辑回归)。对于指数损失函数exponential相当于AdaBoost算法。
  - **回归：**可以选择均方差，绝对损失或者Huber。Huber是均方差和绝对损失的结合。
- **树的最大深度：**树融合的最大深度。最大深度限制了树节点的数量。此参数的优化值依赖于输入变量。可以用逗号隔开的列表指定多个值，这将会增加训练时长。

此算法也提供特征的部分依赖可视化图表。

### (分类与回归) XGBoost

XGBoost使用独立的算法库。

XGBoost是一种高级的梯度提升树算法。它支持并行处理、正则化、early stopping，这使得算法处理速度、扩展性和准确度方面更有优势。

参数：

- **树的最大数量：**XGBoost有early stop机制，这可以优化具体树的数量。更大的实际树的数量会增加训练和预测时间。通常设置：100 - 10000
- **Early stopping：**使用XGBoost内置的early stop机制可以优化具体树的数量。在训练和验证Tab中定义的交叉验证将会被用到。
- **Early stopping轮数：**

Early stopping rounds: The optimizer stops if the loss never decreases for this consecutive number of iterations. Typical values: 1 - 100
Maximum depth of tree: Maximum depth of each tree. High values can increase the quality of the prediction, but can lead to overfitting. Typical values: 3 - 10. You can try multiple values by providing a comma-separated list. This increases the training time.
Learning rate: Lower values slow down convergence and can make the model more robust. Typical values: 0.01 - 0.3. You can try multiple values by providing a comma-separated list. This increases the training time.
L2 regularization: L2 regularization reduces the size of the coefficient for each feature. You can try multiple values by providing a comma-separated list. This increases the training time.
L1 regularization: In addition to reduce overfitting, may improve scoring speed for very high dimensional datasets. You can try multiple values by providing a comma-separated list. This increases the training time.
Gamma: Minimum loss reduction to split a leaf. You can try multiple values by providing a comma-separated list. This increases the training time.
Minimum child weight: Minimum sum of weights(hessian) in a node. High values can prevent overfitting by learning highly specific cases. Smaller values allow leaf nodes to match a small set of rows, which can be relevant for highly imbalanced sets. You can try multiple values by providing a comma-separated list. This increases the training time.
Subsample: Subsample ratio for the data to be used in each tree. Low values can prevent overfitting but can make specific cases harder to learn. Typical values: 0.5 - 1. You can try multiple values by providing a comma-separated list. This increases the training time.
Colsample by tree: Fraction of the features to be used in each tree. Typical values: 0.5-1. You can try multiple values by providing a comma-separated list. This increases the training time.
Replace missing values:
Parallelism: Number of cores used for parallel training. Using more cores leads to faster training but at the expense of more memory consumption, especially for large training datasets. (-1 means “all cores”)
(Regression & Classification) Decision Tree
Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

Parameters:

Maximum depth: The maximum depth of the tree. You can try several values by using a comma separated list. This increases the training time.
Criterion (Gini or Entropy): The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. This applies only to classification problems.
Minimum samples per leaf: Minimum number of samples required to be at a leaf node. You can try several values by using a comma separated list. This increases the training time.
Split strategy (Best or random). The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
(Regression & Classification) Support Vector Machine
Support Vector Machine is a powerful ‘black-box’ algorithm for classification. Through the use of kernel functions, it can learn complex non-linear decision boundaries (ie, when it is not possible to compute the target as a linear combination of input features). SVM is effective with large number of features. However, this algorithm is generally slower than others.

Parameters:

Kernel (linear, RBF, polynomial, sigmoid): The kernel function used for computing the similarity of samples. Try several to see which works the best.
C: Penalty parameter C of the error term. A low value of C will generate a smoother decision boundary (higher bias) while a high value aims at correctly classifying all training examples, at the risk of overfitting (high variance). (C corresponds to the inverse of a regularization parameter). You can try several values of C by using a comma-separated list.
Gamma: Kernel coefficient for RBF, polynomial and sigmoid kernels. Gamma defines the ‘influence’ of each training example in the features space. A low value of gamma means that each example has ‘far-reaching influence’, while a high value means that each example only has close-range influence. If no value is specified (or 0.0), then 1/nb_features is used. You can try several values of Gamma by using a comma-separated list.
Tolerance: Tolerance for stopping criterion.
Maximum number of iterations: Number of iterations when fitting the model. -1 can be used to specific no limit.
(Regression & Classification) Stochastic Gradient Descent
SGD is a family of algorithms that reuse concepts from Support Vector Machines and Logistic Regression. SGD uses an optimized method to minimize the cost (or loss ) function, making it particularly suitable for large datasets (or datasets with large number of features).

Parameters:

Loss function (logit or modified Huber): Selecting ‘logit’ loss will make the SGD behave like a Logistic Regression. Enabling ‘modified huber’ loss will make the SGD behave quite like a Support Vector Machine.
Iterations: number of iterations on the data
Penalty (L1, L2 or elastic net): L1 and L2 regularization are similar to those for linear and logistic regression. Elastic net regularization is a combination of L1 and L2 regularization.
Alpha: Regularization parameter. A high value of alpha (ie, more regularization) will generate a smoother decision boundary (higher bias) while a lower value (less regularization) aims at correctly classifying all training examples, at the risk of overfitting (high variance). You can try several values of alpha by using a comma-separated list.
L1 ratio: ElasticNet regularization mixes both L1 and L2 regularization. This ratio controls the proportion of L2 in the mix. (ie: 0 corresponds to L2-only, 1 corresponds to L1-only). Defaults to 0.15 (85% L2, 15% L1).
Parallelism: Number of cores used for parallel training. Using more cores leads to faster training but at the expense of more memory consumption, especially for large training datasets.
(Regression & Classification) K Nearest Neighbors
K Nearest Neighbor classification makes predictions for a sample by finding the k nearest samples and assigning the most represented class among them.

Warning: this algorithm requires storing the entire training data into the model. This will lead to a very large model if the data is larger than a few hundred lines. Predictions may also be slow.

Parameters:

K: The number of neighbors to examine for each sample. You can try several values by using a comma separated list. This increases the training time.
Distance weighting: If enabled, voting across neighbors will be weighed by the inverse distance from the sample to the neighbor.
Neighbor finding algorithm: The method used to find the nearest neighbors to each point. Has no impact on predictive performance, but will have a high impact on training and prediction speed.
Automatic: a method will be selected empirically depending on the data.
KD & Ball Tree : stores the data points into a partitioned data structure for efficient lookup.
Brute force: will examine every training sample for every prediction. Usually inefficient.
p: The exponent of the Minkowski metric used to search neighbors. For p = 2, this gives Euclidian distance, for p = 1, Manhattan distance. Greater values lead to the Lp distances.
(Regression & Classification) Extra Random Trees
Extra trees, just like Random Forests, are an ensemble model. In addition to sampling features at each stage of splitting the tree, it also samples random threshold at which to make the splits. The additional randomness may improve generalization of the model.

Parameters:

Numbers of trees: Number of trees in the forest. You can try several values by using a comma separated list. This increases the training time.
Feature sampling strategy: Adjusts the number of features to sample at each split.
Automatic will select 30% of the features.
Square root and Logarithm will select the square root or base 2 logarithm of the number of features respectively
Fixed number will select the given number of features
Fixed proportion will select the given proportion of features
Maximum depth of tree: Maximum depth of each tree in the forest. Higher values generally increase the quality of the prediction, but can lead to overfitting. High values also increase the training and prediction time. Use 0 for unlimited depth (ie, keep splitting the tree until each node contains a single target value). You can try several values by using a comma separated list. This increases the training time.
Minimum samples per leaf: Minimum number of samples required in a single tree node to split this node. Lower values increase the quality of the prediction (by splitting the tree mode), but can lead to overfitting and increased training and prediction time. You can try several values by using a comma separated list. This increases the training time.
Parallelism: Number of cores used for parallel training. Using more cores leads to faster training but at the expense of more memory consumption, especially for large training datasets.
(Regression & Classification) Artificial Neural Network
Neural Networks are a class of parametric models which are inspired by the functioning of neurons. They consist of several “hidden” layers of neurons, which receive inputs and transmit them to the next layer, mixing the inputs and applying non-linearities, allowing for a complex decision function.

Parameters:

Hidden layer sizes: Number of neurons on each hidden layer. Separate by commas to add additional layers.
Activation: The activation function for the neurons in the network.
Alpha: L2 regularization parameter. Higher values lead to smaller neuron weights and a more generalizable, although less sharp model.
Max iterations: Maximum iterations for learning. Higher values lead to better convergence, but take more time.
Convergence tolerance: If the loss does not improve by this ratio over two iterations, training stops.
Early stopping: Whether the model should use validation and stop early.
Solver: The solver to use for optimization. LBFGS is a batch algorithm and is not suited for larger datasets.
Shuffle data: Whether the data should be shuffled between epochs (recommended, unless the data is already in random order).
Initial Learning Rate: The initial learning rate for gradient descent.
Automatic batching: Whether batches should be created automatically (will use 200, or the whole dataset if there are less samples). Uncheck to select batch size.
beta_1: beta_1 parameter for ADAM solver.
beta_2: beta_2 parameter for ADAM solver.
epsilon: epsilon parameter for ADAM solver.
(Regression & Classification) Lasso Path
The Lasso Path is a method which computes the LASSO path (ie. for all values of the regularization parameter). This is performed using LARS regression. It requires a number of passes on the data equal to the number of features. If this number is large, computation may be slow. This computation allows to select a given number of non-zero coefficients, ie. to select a given number of features. After training, you will be able to visualize the LASSO path and select a new number of features.

Parameters:

Maximum features: The number of kept features. Input 0 to have all features enabled (no regularization). Has no impact on training time.
(Regression & Classification) Custom Models
You can also specify custom models using Python.

Your custom models should follow the scikit-learn predictor protocol with proper fit and predict methods.

Code samples are available for custom models.

Clustering algorithms
K-means
The k-means algorithm clusters data by trying to separate samples in n groups, minimizing a criterion known as the ‘inertia’ of the groups.

Parameters:

Number of clusters: You can try multiple values by providing a comma-separated list. This increases the training time.
Seed: Used to generate reproducible results. 0 or no value means that no known seed is used (results will not be fully reproducible)
Parallelism: Number of cores used for parallel training. Using more cores leads to faster training but at the expense of more memory consumption. If -1 all CPUs are used. For values below -1, (n_cpus + 1 + value) are used: ie for -2, all CPUs but one are used.
Gaussian Mixture
The Gaussian Mixture Model models the distribution of the data as a “mixture” of several populations, each of which can be described by a single multivariate normal distribution.

An example of such a distribution is that of sizes among adults, which is described by the mixture of two distributions: the sizes of men, and those of women, each of which is approximately described by a normal distribution.

Parameters:

Number of mixture components: Number of populations. You can try multiple values by providing a comma-separated list. This increases the training time.
Max Iterations: The maximum number of iterations to learn the model. The Gaussian Mixture model uses the Expectation-Maximization algorithm, which is iterative, each iteration running on all of the data. A higher value of this parameter will lead to a longer running time, but a more precise clustering. A value between 10 and 100 is recommended.
Seed: Used to generate reproducible results. 0 or no value means that no known seed is used (results will not be fully reproducible)
Mini-batch K-means
The Mini-Batch k-means is a variant of the k-means algorithm which uses mini-batches to reduce the computation time, while still attempting to optimise the same objective function.

Parameters:

Numbers of clusters: You can try multiple values by providing a comma-separated list. This increases the training time.
Seed: Used to generate reproducible results. 0 or no value means that no known seed is used (results will not be fully reproducible)
Agglomerative Clustering
Hierarchical clustering is a general family of clustering algorithms that build nested clusters by merging them successively. This hierarchy of clusters represented as a tree (or dendrogram). The root of the tree is the unique cluster that gathers all the samples, the leaves being the clusters with only one sample.

Parameters:

Numbers of clusters: You can try multiple values by providing a comma-separated list. This increases the training time.
Spectral Clustering
Spectral clustering algorithm uses the graph distance in the nearest neighbor graph. It does a low-dimension embedding of the affinity matrix between samples, followed by a k-means in the low dimensional space.

Parameters:

Numbers of clusters: You can try several values by using a comma-separated list. This increases the training time.
Affinity measure: The method to computing the distance between samples. Possible options are nearest neighbors, RBF kernel and polynomial kernel.
Gamma: Kernel coefficient for RBF and polynomial kernels. Gamma defines the ‘influence’ of each training example in the features space. A low value of gamma means that each example has ‘far-reaching influence’, while a high value means that each example only has close-range influence. If no value is specified (or 0.0), then 1/nb_features is used.
Coef0: Independent term for ‘polynomial’ or ‘sigmoid’ kernel function.
Seed: Used to generate reproducible results. 0 or no value means that no known seed is used (results will not be fully reproducible)
DBSCAN
The DBSCAN algorithm views clusters as areas of high density separated by areas of low density. Due to this rather generic view, clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters are convex shaped. Numerical features should use standard rescaling.

There are two parameters that you can modify in DBSCAN:

Epsilon: Maximum distance to consider two samples in the same neighborhood. You can try several values by using a comma-separated list
Min. Sample ratio: Minimum ratio of records to form a cluster
Interactive Clustering (Two-step clustering)
Interactive clustering is based on a two-step clustering algorithm. This two-staged algorithm first agglomerates data points into small clusters using K-Means clustering. Then, it applies agglomerative hierarchical clustering in order to further cluster the data, while also building a hierarchy between the smaller clusters, which can then be interpreted. It therefore allows to extract hierarchical information from datasets larger than a few hundred lines, which cannot be achieved through standard methods. The clustering can then be manually adjusted in DSS’s interface.

Parameters:

Number of Pre-clusters: The number of clusters for KMeans preclustering. It is recommended that this number be lower than a couple hundred for readability.
Number of clusters: The number of clusters in the hierarchy. The full hierarchy will be built and displayed, but these clusters will be used for scoring.
Max Iterations: The maximum number of iterations for preclustering. KMeans is an iterative algorithm. A higher value of this parameter will lead to a longer running time, but a more precise pre-clustering. A value between 10 and 100 is recommended.
Seed: Used to generate reproducible results. 0 or no value means that no known seed is used (results will not be fully reproducible)
Isolation Forest (Anomaly Detection)
Isolation forest is an anomaly detection algorithm. It isolates observations by creating a Random Forest of trees, each splitting samples in different partitions. Anomalies tend to have much shorter paths from the root of the tree. Thus, the mean distance from the root provides a good measure of non-normality.

Parameters:

Number of trees: Number of trees in the forest.
Contamination: Expected proportion of anomalies in the data.
Anomalies to display: Maximum number of anomalies to display in the model report. Too high a number may cause memory and UI problems.
Custom Models
You can also specify custom models using Python.

Your custom models should follow the scikit-learn predictor protocol with proper fit and fit_predict methods.

A specified number of clusters can also be passed to the model through the interface.

Code samples are available for custom models.