# End to end data science technical process

## Data engineering and Data Science

## Code

```
import pandas as pd
import numpy as np
```

```
df = pd.read_csv('LWR_BSA_Notebooks/Out/csv/BSAClean2021-04-23_16-40-46.csv')
```

```
df.head()
```

```
df["Ball Class_BSADial"].value_counts()
```

```
boxplot = df.boxplot(column=['Ball Class_BSADial'])
```

```
ax1 = df.plot.scatter(x='Ball Class_BSADial',
                      y='Rack Dia._BSADial',
                      c='DarkBlue')
```

```
ax1 = df.plot.scatter(x='Ball Class_BSADial',
                      y='Nest No._BSADial',
                      c='DarkBlue')
```

```
ax1 = df.plot.scatter(x='Ball Class_BSADial',
                      y='Nut Dia._BSADial',
                      c='DarkBlue')
```

```
ax1 = df.plot.scatter(x='Ball Class_BSADial',
                      y='Offset_BNG',
                      c='DarkBlue')
```

```
ax1 = df.plot.scatter(x='Ball Class_BSADial',
                      y='AXIAL_BNOP59',
                      c='DarkBlue')
```

```
from pandas_profiling import ProfileReport
import seaborn as sns
from fairlearn.reductions import DemographicParity
from fairlearn.metrics import selection_rate
import matplotlib.pyplot as plt
```

```
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['AXIAL_BNOP59'], df['Ball Class_BSADial'])
ax.set_xlabel('AXIAL_BNOP59')
ax.set_ylabel('Ball Class_BSADial')
plt.show()
```

```
profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
```

```
profile.to_widgets()
```

```
profile.to_notebook_iframe()
```

```
df.describe()
```

```
import seaborn as sns
sns.boxplot(x=df['Ball Class_BSADial'])
```

```
corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
```

```
covar = df.cov()
ax = sns.heatmap(
    covar, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
```

## modelling

```
from sklearn.model_selection import train_test_split
```

```
y = df["Ball Class_BSADial"]
```

```
X = df.drop(columns="Ball Class_BSADial")
```

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

```
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

```
from sklearn.metrics import classification_report
```

```
print(classification_report(y_test, y_pred))
```

```
from sklearn.metrics import accuracy_score
```

```
accuracy_score(y_test, y_pred)
accuracy_score(y_test, y_pred, normalize=False)
```

```
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
roc_auc_score(y_pred,y_test)
```

```
from sklearn.inspection import permutation_importance
r = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=0)
```

```
for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
         print(f"{df.columns[i]:<8}"
               f"{r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")
```

```
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
```

```
eclf1 = VotingClassifier(estimators=[
         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf1 = eclf1.fit(X_train, y_train)
print(eclf1.predict(X_test))
```

```
import lightgbm as lgb
```

```
d_train=lgb.Dataset(X_train, label=y_train)
```

```
#Specifying the parameter
params={}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['max_depth']=10
params['num_class']=13
```

```
#train the model 
clf=lgb.train(params,d_train,100) #train the model on 100 epocs
```

```
#prediction on the test set
y_pred=clf.predict(X_test)
```

```
print(y_pred)
```

```
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
```

```
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
```

```
bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")

bdt_real.fit(X_train, y_train)
bdt_discrete.fit(X_train, y_train)

real_test_errors = []
discrete_test_errors = []

for real_test_predict, discrete_train_predict in zip(
        bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
    real_test_errors.append(
        1. - accuracy_score(real_test_predict, y_test))
    discrete_test_errors.append(
        1. - accuracy_score(discrete_train_predict, y_test))

n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)
```

```
# Boosting might terminate early, but the following arrays are always
# n_estimators long. We crop them to the actual number of trees here:
discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(range(1, n_trees_discrete + 1),
         discrete_test_errors, c='black', label='SAMME')
plt.plot(range(1, n_trees_real + 1),
         real_test_errors, c='black',
         linestyle='dashed', label='SAMME.R')
plt.legend()
plt.ylim(0.18, 0.62)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')

plt.subplot(132)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
         "b", label='SAMME', alpha=.5)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
         "r", label='SAMME.R', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,
         max(real_estimator_errors.max(),
             discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(133)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
         "b", label='SAMME')
plt.legend()
plt.ylabel('Weight')
plt.xlabel('Number of Trees')
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))

# prevent overlapping y-axis labels
plt.subplots_adjust(wspace=0.25)
plt.show()
```

```
def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v
```

```
df.corr(method=histogram_intersection)
```