---
title: "kaggle House Priceã«ææ¦!"
emoji: "ð¤"
type: "tech" # tech: æè¡è¨äº / idea: ã¢ã¤ãã¢
topics: ["python", "æ©æ¢°å­¦ç¿", "åå¿è", "ãã­ã°ã©ãã³ã°"]
published: true
---
# åãã«
ãã®è¨äºã§ã¯ãkaggleã®House Priceã³ã³ãã«ã¤ãã¦èª¬æãã¦ããã¾ãï¼ãªããã³ã¼ãã¯ä»¥ä¸ãåèã«ãã¦ä¸ããã
https://www.kaggle.com/code/iketuba/house-price-2022-06-24

# ã©ã¤ãã©ãªã®èª­ã¿è¾¼ã¿
ã¾ããå¿è¦ãªã©ã¤ãã©ãªã®èª­ã¿è¾¼ã¿ãè¡ãã¾ãã
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
```

# ãã¼ã¿ã®èª­ã¿è¾¼ã¿
ãã¼ã¿ã®èª­ã¿è¾¼ã¿ãè¡ãã¾ãã
```py
# å­¦ç¿ãã¼ã¿ããã¹ããã¼ã¿ãæåºãµã³ãã«ãã¼ã¿ã®èª­ã¿è¾¼ã¿
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
```

ãã¼ã¿ã®å¤§ãããç¢ºèªãã¾ãã
```py
# ãã¼ã¿ãµã¤ãºã®ç¢ºèª
print('å­¦ç¿ãã¼ã¿ã®ãµã¤ãº:', train.shape)
print('ãã¹ããã¼ã¿ã®ãµã¤ãº:', test.shape)
```

ãã¼ã¿ã®æ¬ æå¤ãåãç¢ºèªãã¾ãã
```py
# ãã¼ã¿ã®æå ±ã®ç¢ºèª
print(train.info())
print("-"*10)
print(test.info())
```

ããããããã¼ã¿ã«ã¯æ¬ æå¤ãå¤ãå«ã¾ãããã¨ãåããã¾ããããã«ããã¼ã¿ã®åãobjectã¨ãªã£ã¦ããå¤æ°ãå¤ããããæ°å¤å¤æ°ã«å¤æããå¿è¦ãããã¾ãã<br>
æ¬¡ã«ãå­¦ç¿ç¨ãã¼ã¿ãèª¬æå¤æ°ã¨ç®çå¤æ°ã«åãã¾ãã
```py
# å­¦ç¿ç¨ãã¼ã¿ãç¹å¾´éã¨ç®çå¤æ°ã«åãã
train_x = train.drop(['SalePrice'], axis=1)
train_y = train['SalePrice']

# ãã¹ããã¼ã¿ã¯ç¹å¾´éã®ã¿ãªã®ã§ãã®ã¾ã¾ã§è¯ã
test_x = test.copy()
```

# ç¹å¾´éã®ä½æ
ã¾ãããã¼ã¿ã®æ¬ æå¤ãç¢ºèªãã¾ãã
```py
#å­¦ç¿ãã¼ã¿ã®æ¬ æå¤ãç¢ºèªãã
print('è¨ç·´ãã¼ã¿ã®æ¬ æå¤:\n', train_x.isnull().sum().sort_values(ascending=False))
print('è¨ç·´ãã¼ã¿ã®æ¬ æå¤(å²å):\n', 
      (train_x.isnull().sum() / len(train_x)).sort_values(ascending=False))
print("-"*10)
#ãã¹ããã¼ã¿ã®æ¬ æå¤ãç¢ºèªãã
print('ãã¹ããã¼ã¿ã®æ¬ æå¤:\n', test_x.isnull().sum().sort_values(ascending=False))
print('ãã¹ããã¼ã¿ã®æ¬ æå¤(å²å):\n', 
      (test_x.isnull().sum() / len(test_x)).sort_values(ascending=False))
```

å¤æ°Idã¯çªå·ãæ¯ã£ã¦ããã ãã§ãããå®¶ã®ä¾¡æ ¼ã«å½±é¿ãåã¼ããªãã¨èããããã®ã§ãé¤å¤ãã¾ãã
```py
# å¤æ°Idãé¤å¤ãã
train_x = train_x.drop(['Id'], axis=1)
test_x = test_x.drop(['Id'], axis=1)
```

æ¬¡ã«ãæ¬ æå¤ã®å¦çããã¾ããå­¦ç¿ç¨ãã¼ã¿ã¨ãã¹ããã¼ã¿ã«åãå¦çãããããã2ã¤ã®ãã¼ã¿ãçµåãã¾ããããããæ¹ãã³ã¼ããç­ãã¦æ¸ã¿ã¾ãã
```py
# å­¦ç¿ãã¼ã¿ã¨ãã¹ããã¼ã¿ãçµåãã
all_x = pd.concat([train_x, test_x])
```

æ¬ æå¤ãå¤ãããå¤æ°ã¯åé¤ãããã¨æãã¾ããããã§ã¯ãæ¬ æå¤ãååä»¥ä¸å ããå¤æ°ã¯åé¤ãã¾ãã(ååã«ããã®ã¯ãªãã¨ãªãã§ã)
```py
# æ¬ æå¤ãååä»¥ä¸å ããç¹å¾´éã¯é¤å¤ãã
miss_cols = []

for col in all_x.columns:
    if all_x[col].isnull().sum()/len(all_x) > 0.5:
        miss_cols.append(col)
        
all_x = all_x.drop(miss_cols, axis=1)
```

åé¤ããªãã£ãå¤æ°ã®ä¸­ã§ãæ¬ æã®ããå¤æ°ã«ã¤ãã¦å¦çãè¡ãã¾ããæ°å¤å¤æ°ã«ã¤ãã¦ã¯ä¸­å¤®å¤ã§è£å®ãããã¨æãã¾ãã
```py
# æ°å¤å¤æ°ãã¤æ¬ æå¤ãå«ãç¹å¾´éãä¸­å¤®å¤ã§è£å®ãã
# æ°å¤å¤æ°ãã¤æ¬ æå¤ãå«ãç¹å¾´éãæãåºã
num_miss_cols = []

for col in all_x.columns:
    if all_x[col].dtype in ['int64', 'float64'] and all_x[col].isnull().sum() > 0:
        num_miss_cols.append(col)
        
# æ¬ æå¤ãä¸­å¤®å¤ã§è£å®ãã
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
imp_mean.fit(all_x[num_miss_cols])
all_x[num_miss_cols] = imp_mean.transform(all_x[num_miss_cols])
```

ã«ãã´ãªå¤æ°ã«ã¤ãã¦ã¯æé »å¤ã§è£å®ãããã¨æãã¾ãã
```py
# ã«ãã´ãªå¤æ°ãã¤æ¬ æå¤ãå«ãåãæé »å¤ã§è£å®ãã
# ã«ãã´ãªå¤æ°ãã¤æ¬ æå¤ãå«ãç¹å¾´éãæãåºã
cat_miss_cols = []

for col in all_x.columns:
    if all_x[col].dtype == 'object' and all_x[col].isnull().sum() > 0:
        cat_miss_cols.append(col)
        
# æ¬ æå¤ãæé »å¤ã§è£å®ãã
from sklearn.impute import SimpleImputer
imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_most_frequent.fit(all_x[cat_miss_cols])
all_x[cat_miss_cols] = imp_most_frequent.transform(all_x[cat_miss_cols])
```

æ¬ æå¤ã®å¦çãã§ããã®ã§ãçµåãããã¼ã¿ãå­¦ç¿ç¨ãã¼ã¿ã¨ãã¹ããã¼ã¿ã«ååå²ãã¾ãã
```py
# å­¦ç¿ãã¼ã¿ã¨ãã¹ããã¼ã¿ã«ååå²ãã
train_x = all_x.iloc[:train_x.shape[0], :]
test_x = all_x.iloc[train_x.shape[0]:, :]
```

æ¬¡ã«ãã«ãã´ãªå¤æ°ãæ°å¤å¤æ°ã«å¤æãã¦ããã¾ããããã§ã¯ãã©ãã«ã¨ã³ã³ã¼ãã£ã³ã°ãã¾ãã
```py
# è³ªçå¤æ°ãã©ãã«ã¨ã³ã³ã¼ãã£ã³ã°ãã
# è³ªçå¤æ°ãæãåºã
cat_cols = []

for col in train_x.columns:
    if train_x[col].dtype == 'object':
        cat_cols.append(col)
        
# å­¦ç¿ãã¼ã¿ã¨ãã¹ããã¼ã¿ãçµåãã¦ã©ãã«ã¨ã³ã³ã¼ãã£ã³ã°ãã
from sklearn.preprocessing import LabelEncoder
all_x = pd.concat([train_x, test_x])
for col in cat_cols:
    le = LabelEncoder()
    all_x[col] = le.fit_transform(all_x[col])

# å­¦ç¿ãã¼ã¿ã¨ãã¹ããã¼ã¿ã«ååå²
train_x = all_x.iloc[:train_x.shape[0], :]
test_x = all_x.iloc[train_x.shape[0]:, :]
```

ããã§ãåå¦çãå®äºãã¾ãããæå¾ã«æ¨æºåããã¦ããã¾ãã
```py
# æ¨æºå
scaler = preprocessing.StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)
```

# ã¢ãã«ã®ä½æã¨è©ä¾¡ãæåº
åå²æ¹æ³ãæå®ãã¾ãã
```py
# åå²æ¹æ³ã®æå®
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
```

äº¤å·®æ¤è¨¼æ³ãç¨ãã¦ãç²¾åº¦ãç¢ºèªãã¾ãã
```py
# ã¢ãã«ã®ä½æã¨è©ä¾¡(RandomForestRegressor)
rfr = ensemble.RandomForestRegressor()
rfr_results = model_selection.cross_validate(rfr, train_x, train_y, scoring='neg_root_mean_squared_error',cv=kf)
rfr_results['test_score'].mean()
```

ã¢ãã«ã®ãã¥ã¼ãã³ã°ãè¡ãã¾ããGridSearchCVãç¨ãã¾ããã
```py
# ã¢ãã«ã®ä½æã¨è©ä¾¡ - ãã¥ã¼ãã³ã°ãã(RandomForestRegressor)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 2, 5],
}

tune_rfr = model_selection.GridSearchCV(rfr, param_grid=param_grid, scoring = 'neg_root_mean_squared_error', cv = kf)
                                                                   
tune_rfr.fit(train_x, train_y)
print("æããããã©ã¡ã¼ã¿: ", tune_rfr.best_params_)
print("æ¤è¨¼ãã¼ã¿ã®å¹³åå¤: ", tune_rfr.cv_results_['mean_test_score'][tune_rfr.best_index_])
```

ãã¥ã¼ãã³ã°ã®çµæãæãç²¾åº¦ã®é«ãã£ãã¢ãã«ãç¨ãã¾ããå­¦ç¿ç¨ãã¼ã¿å¨ä½ã§ã¢ãã«ã®å­¦ç¿ããã¦ããã¹ããã¼ã¿ã«å¯¾ãã¦äºæ¸¬ãè¡ãã¾ãã
```py
# å­¦ç¿ãã¼ã¿å¨ä½ã§ã¢ãã«ã®å­¦ç¿ããã
tune_rfr.best_estimator_.fit(train_x, train_y)

# ãã¹ããã¼ã¿ã«å¯¾ãã¦äºæ¸¬ãã
predict = tune_rfr.best_estimator_.predict(test_x)
```

æå¾ã«æåºãã¡ã¤ã«ãä½æãã¾ãã
```py
# æåºç¨ãã¡ã¤ã«ã®ä½æ
submit_1 = pd.DataFrame({'Id': test['Id'], 'SalePrice': predict})
submit_1.to_csv('submission.csv', index=False)
```

# æå¾ã«
ä»åã¯åå¤æ°ã«ã¤ãã¦ç´°ããåæãè¡ã£ã¦ããªãã®ã§ãããå°ãæ·±ãåæãããè¨äºãæ¸ãããã¨æã£ã¦ã¾ãï¼