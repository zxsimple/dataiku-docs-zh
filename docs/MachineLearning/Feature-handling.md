特征处理

> 注意：在模型 > 设置 > 特征Tab中对特征处理设置
>
> 

Note

You can change the settings for feature processing under Models > Settings > Features tab

Most [machine learning engines](https://doc.dataiku.com/dss/latest/machine_learning/engines.html) in DSS visual machine learning can only process numerical features, with no missing values.

DSS allows users to specify pre-processing of variables before model training.

## Roles

A feature’s role determines how it’s used during machine learning.

- **Reject** means that the feature is not used
- **Input** means that the feature is used to build a model, either as a potential [predictor for a target](https://doc.dataiku.com/dss/latest/machine_learning/supervised.html)) or for [clustering](https://doc.dataiku.com/dss/latest/machine_learning/unsupervised.html)
- **Use for display only** means that the feature is not used to build a model, but is used to label model output. This role is currently only used by cluster models.

## Variable type

A feature’s variable type determines the feature handling options during machine learning.

- **Categorical** variables take one of an enumerated list values. The goal of categorical feature handling is to encode the values of a categorical variable so that they can be treated as numeric.
- **Numerical** variables take values that can be added, subtracted, multiplied, and so on. There are times when it may be useful to treat a numerical variable with a limited number of values as categorical.
- **Text** variables are arbitrary blocks of text. If a text variable takes a limited number of values, it may be useful to treat it as categorical.

### Categorical variable handling

The **Category handling** and **Missing values** methods, and their related controls, specify how a categorical variable is handled.

- **Dummy-encoding (vectorization)** creates a vector of 0/1 flags of length equal to the number of categories in the categorical variable. You can choose to drop one of the dummies so that they are not linearly dependent, or let Dataiku decide. There is a limit on the number of dummies, which can be based on a maximum number of categories, the cumulative proportion of rows accounted for by the most popular rows, or a minimum number of samples per category.
- **Replace by 0/1 flag indicating presence**
- **Impact-coding**
- **Feature hashing (for high cardinality)**

### Numerical variable handling

The **Numerical handling** and **Missing values** methods, and their related controls, specify how a numerical variable is handled.

- **Keep as a regular numerical feature** allows for rescaling prior to training, which can improve model performance in some instances. Standard rescaling scales the feature to a standard deviation of one and a mean of zero. Min-max rescaling sets the minimum value of the feature to zero and the max to one. In addition, post-rescaling, you can request that derived features such as sqrt(x), x^2, … be generated and considered in the model. *Rescale numeric variables if there are large differences in the absolute values of the features.*
- **Replace by 0/1 flag indicating presence**
- **Binarize based on a threshold** replaces the feature values with a 0/1 flag that indicates whether the value is above or below the specified threshold.
- **Quantize** replaces the feature values with the quantiles of the feature’s empirical distribution.

### Text variable handling

### Missing values

There are a few choices for handling missing values in categorical and numerical features.

- **Treat as a regular value** (categorical features only) treats missing values as a distinct category. This should be used for **structurally missing** data that are impossible to measure, e.g. the US state for an address in Canada.
- **Impute…** replaces missing values with the specified value. This should be used for **randomly missing** data that are missing due to random noise.
- **Drop rows** discards rows with missing values from the model building. *Avoid discarding rows, unless missing data is extremely rare*.

## Custom Preprocessings

DSS allows to define custom python preprocessings, in order to plug user-generated code which will process a feature. This is done by selection “Custom preprocessing” in the feature handling options. The way to do this is to implement a class with two methods :

```
def fit(self, series):
def transform(self, series):
```

Here, series is a pandas Series object representing the feature column. The fit method does not need to return anything, but must modify the object in-place if fitting is necessary. The transform method must return either a pandas DataFrame or a 2-D numpy array or scipy.sparse.csr_matrix containing the preprocessed result. Note that a single processor may output several numerical features, corresponding several columns of the output. If a numpy array or scipy.sparse.csr_matrix is chosen, then the processor should be also have a “names” attribute, containing the list of the output feature names.

To use your processor in the visual ML UI, you must import it and instantiate it in the code editor, by assigning the processor to the “processor” variable, as follows :

```
from mymodule import MyProcessor
processor = MyProcessor()
```

As with any python code component, classes must be defined in a file stored in the lib/python folder of the data directory.