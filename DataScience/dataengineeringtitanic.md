`# Data Engineering process for Data Science

## How to do data engineering to validate data before modelling

## Code

- First load the imports

```
from azureml.core import Workspace, Dataset
from raiwidgets import ExplanationDashboard
from azureml.core import Dataset
from azureml.data.dataset_factory import DataType
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
```

- now time to load the data 

```
df = pd.read_csv('Titanic.csv')
df.head()
```

- Now work on insights on data by data profiling
- first mean

```
df.mean()
```

- Next is median

```
df.median()
```

- Max values

```
df.max()
```

- Now load the pandas profiling

```
profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
```

```
profile.to_widgets()
```

```
profile.to_notebook_iframe()
```

- Run stats on dataframe

```
df.describe()
```

- Now lets do box plots to find outliers

```
import seaborn as sns
sns.boxplot(x=df['Survived'])
```

```
sns.boxplot(x=df['Age'])
```

- Plot scatter plot chart
- This allows us to see correlation

```
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['Age'], df['Survived'])
ax.set_xlabel('Age')
ax.set_ylabel('Survived')
plt.show()
```

```
df.corr(method ='pearson')
```

```
df.corr()
```

- Get correlation matrix

```
corrMatrix = df.corr()
```

```
import seaborn as sn
import matplotlib.pyplot as plt
```

```
sn.heatmap(corrMatrix, annot=True)
plt.show()
```

```
covMatrix = df.cov()
```

```
sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()
```

- build stats and see

```
import scipy.stats
```

```
scipy.stats.rankdata(df["Survived"])
```

- calculate spearman

```
rho, p = scipy.stats.spearmanr(df["Survived"], df["Age"])
rho
```

```
result = scipy.stats.spearmanr(df["Age"], df["Survived"])
result.correlation
```

```
slope, intercept, r, p, stderr = scipy.stats.linregress(df["Age"], df["Survived"])
```

```
p
```

- Next will more data engineering
- Idea here is to make sure the data doesn't have outliers
- Also make sure data is available for prediction