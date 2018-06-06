Scoring engines
DSS allows you to select various engines in order to perform scoring of your models. This allows for faster execution in some cases.

Note

Scoring engines are only used to actually predict rows. While they are strongly related to training engines, some models trained with one engine can be scored with another.

The following scoring engines are available:

Local (DSS server only) scoring. This engine has two variants: * the Python engine provides wider compatibility but lower performance. * the Optimized scorer provides better performance and is automatically used whenever possible.
Spark: the scoring is performed in a distributed fashion on a Spark cluster
SQL: the model is converted to SQL code and executed within a SQL database.
The selected engine can be adjusted in the scoring recipe editor. Only engines that are compatible with the selected model and input dataset will be available.

The default settings the following:

If the model was trained with Spark MLLib or Sparkling Water, it will be scored with the Spark engine
If the model was trained with VerticaML, it will be scored with the SQL engine
Else it will be scored with the Local engine. The optimized engine will be used if available.
If you do not wish to score your model with the “optimized” engine for some reason, you may select “Force original backend” in the scoring recipe editor to revert to the original backend.

Choosing an SQL engine (if your scored dataset is stored in an SQL database and your model is compatible) will generate a request to score the dataset. Note that this may create very large requests for complex models.

The compatibility matrix for all DSS models is the following:

Training engine	Algorithm	Local (Optimized)	Local (Python)	Spark	SQL	 
Python in-memory	Random forest	Yes	Yes	Yes	Yes (no probas for multiclass)	 
MLLib	Random forest	Yes	Yes	Yes	Yes (no probas for multiclass)	 
Python in-memory	Gradient Boosting	Yes	Yes	Yes	Regression only	 
MLLib	Gradient Boosting	Yes	Yes	Yes	Regression only	 
Python in-memory	Extra Trees (Scikit)	Yes	Yes	Yes	Yes (no probas for multiclass)	 
Python in-memory	Decision Trees	Yes	Yes	Yes	Yes (no probas for multiclass)	 
MLLib	Decision Trees	Yes	Yes	Yes	Yes (no probas for multiclass)	 
Python in-memory	Ordinary Least Squares, Lasso, Ridge	Yes	Yes	Yes	Yes	 
Python in-memory	SGD	Yes	Yes	Yes	Yes	 
MLLib	Linear Regression	Yes	Yes	Yes	Yes	 
Python in-memory	Logistic Regression	Yes	Yes	Yes	Yes	 
MLLib	Logistic Regression	Yes	Yes	Yes	Yes	 
Python in-memory	Neural Networks	Yes	Yes	Yes	Yes	 
Python in-memory	Naive Bayes	No	Yes	No	No	 
MLLib	Naive Bayes	No	No	Yes	No	 
Python in-memory	K-nearest-neighbors	No	Yes	No	No	 
Python in-memory	XGBoost	No	Yes	No	No	 
Python in-memory	SVM	No	Yes	No	No	 
Python in-memory	Custom models	No	Yes	No	No	 
MLLib	Custom models	No	No	Yes	No	 
Sparkling-Water	All models	No	No	Yes	No	 
VerticaML	All models	No	No	No	Yes	 
Note

For models trained with Python, the Optimized Local and Spark engines are only available if preprocessing is also compatible.

The following preprocessing options are available for optimized scoring:

Numerical
No rescaling
Rescaling
Binning
Derivative features
Flag missing
Imputation
Drop row
Categorical
Dummification
Impact coding
Flag missing
Hashing (MLLib only)
Impute
Drop row
Text
Count vectorization
TF/IDF vectorization
Hashing (MLLib)
Note

For all models but VerticaML, the SQL engine is only available if preprocessing is also compatible.

The following preprocessing options are available for SQL scoring :

Numerical
No rescaling
Rescaling
Binning
Derivative features
Flag missing
Imputation
Drop row
Categorical
Dummification
Impact coding
Flag missing
Imputation
Drop row
Text is not supported

Limitations
The following limitations exist with SQL scoring:

Some algorithms may not generate probabilities with SQL scoring (see table above)
Conditional output columns are not generated with SQL scoring
Preparation scripts are not compatible with SQL scoring
Multiclass logistic regression and neural networks require the SQL dialect to support the GREATEST function.