---
title: "kaggle House Priceに挑戦!"
emoji: "🤖"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["python", "機械学習", "初心者", "プログラミング"]
published: true
---
# 初めに
この記事では、kaggleのHouse Priceコンペについて説明していきます！なお、コードは以下を参考にして下さい。
https://www.kaggle.com/code/iketuba/house-price-2022-06-24

# ライブラリの読み込み
まず、必要なライブラリの読み込みを行います。
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

# データの読み込み
データの読み込みを行います。
```py
# 学習データ、テストデータ、提出サンプルデータの読み込み
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
```

データの大きさを確認します。
```py
# データサイズの確認
print('学習データのサイズ:', train.shape)
print('テストデータのサイズ:', test.shape)
```

データの欠損値や型を確認します。
```py
# データの情報の確認
print(train.info())
print("-"*10)
print(test.info())
```

ここから、データには欠損値が多く含まれることが分かります。さらに、データの型がobjectとなっている変数が多くあり、数値変数に変換する必要があります。<br>
次に、学習用データを説明変数と目的変数に分けます。
```py
# 学習用データを特徴量と目的変数に分ける
train_x = train.drop(['SalePrice'], axis=1)
train_y = train['SalePrice']

# テストデータは特徴量のみなのでそのままで良い
test_x = test.copy()
```

# 特徴量の作成
まず、データの欠損値を確認します。
```py
#学習データの欠損値を確認する
print('訓練データの欠損値:\n', train_x.isnull().sum().sort_values(ascending=False))
print('訓練データの欠損値(割合):\n', 
      (train_x.isnull().sum() / len(train_x)).sort_values(ascending=False))
print("-"*10)
#テストデータの欠損値を確認する
print('テストデータの欠損値:\n', test_x.isnull().sum().sort_values(ascending=False))
print('テストデータの欠損値(割合):\n', 
      (test_x.isnull().sum() / len(test_x)).sort_values(ascending=False))
```

変数Idは番号を振っているだけであり、家の価格に影響を及ぼさないと考えられるので、除外します。
```py
# 変数Idを除外する
train_x = train_x.drop(['Id'], axis=1)
test_x = test_x.drop(['Id'], axis=1)
```

次に、欠損値の処理をします。学習用データとテストデータに同じ処理をするため、2つのデータを結合します。こうした方がコードが短くて済みます。
```py
# 学習データとテストデータを結合する
all_x = pd.concat([train_x, test_x])
```

欠損値が多すぎる変数は削除したいと思います。ここでは、欠損値が半分以上占める変数は削除します。(半分にしたのはなんとなくです)
```py
# 欠損値が半分以上占める特徴量は除外する
miss_cols = []

for col in all_x.columns:
    if all_x[col].isnull().sum()/len(all_x) > 0.5:
        miss_cols.append(col)
        
all_x = all_x.drop(miss_cols, axis=1)
```

削除しなかった変数の中で、欠損のある変数について処理を行います。数値変数については中央値で補完したいと思います。
```py
# 数値変数かつ欠損値を含む特徴量を中央値で補完する
# 数値変数かつ欠損値を含む特徴量を抜き出す
num_miss_cols = []

for col in all_x.columns:
    if all_x[col].dtype in ['int64', 'float64'] and all_x[col].isnull().sum() > 0:
        num_miss_cols.append(col)
        
# 欠損値を中央値で補完する
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
imp_mean.fit(all_x[num_miss_cols])
all_x[num_miss_cols] = imp_mean.transform(all_x[num_miss_cols])
```

カテゴリ変数については最頻値で補完したいと思います。
```py
# カテゴリ変数かつ欠損値を含む列を最頻値で補完する
# カテゴリ変数かつ欠損値を含む特徴量を抜き出す
cat_miss_cols = []

for col in all_x.columns:
    if all_x[col].dtype == 'object' and all_x[col].isnull().sum() > 0:
        cat_miss_cols.append(col)
        
# 欠損値を最頻値で補完する
from sklearn.impute import SimpleImputer
imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_most_frequent.fit(all_x[cat_miss_cols])
all_x[cat_miss_cols] = imp_most_frequent.transform(all_x[cat_miss_cols])
```

欠損値の処理ができたので、結合したデータを学習用データとテストデータに再分割します。
```py
# 学習データとテストデータに再分割する
train_x = all_x.iloc[:train_x.shape[0], :]
test_x = all_x.iloc[train_x.shape[0]:, :]
```

次に、カテゴリ変数を数値変数に変換していきます。ここでは、ラベルエンコーディングします。
```py
# 質的変数をラベルエンコーディングする
# 質的変数を抜き出す
cat_cols = []

for col in train_x.columns:
    if train_x[col].dtype == 'object':
        cat_cols.append(col)
        
# 学習データとテストデータを結合してラベルエンコーディングする
from sklearn.preprocessing import LabelEncoder
all_x = pd.concat([train_x, test_x])
for col in cat_cols:
    le = LabelEncoder()
    all_x[col] = le.fit_transform(all_x[col])

# 学習データとテストデータに再分割
train_x = all_x.iloc[:train_x.shape[0], :]
test_x = all_x.iloc[train_x.shape[0]:, :]
```

これで、前処理が完了しました。最後に標準化をしておきます。
```py
# 標準化
scaler = preprocessing.StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)
```

# モデルの作成と評価、提出
分割方法を指定します。
```py
# 分割方法の指定
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
```

交差検証法を用いて、精度を確認します。
```py
# モデルの作成と評価(RandomForestRegressor)
rfr = ensemble.RandomForestRegressor()
rfr_results = model_selection.cross_validate(rfr, train_x, train_y, scoring='neg_root_mean_squared_error',cv=kf)
rfr_results['test_score'].mean()
```

モデルのチューニングを行います。GridSearchCVを用いました。
```py
# モデルの作成と評価 - チューニングあり(RandomForestRegressor)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 2, 5],
}

tune_rfr = model_selection.GridSearchCV(rfr, param_grid=param_grid, scoring = 'neg_root_mean_squared_error', cv = kf)
                                        
                                        
tune_rfr.fit(train_x, train_y)
print("最もよいパラメータ: ", tune_rfr.best_params_)
print("検証データの平均値: ", tune_rfr.cv_results_['mean_test_score'][tune_rfr.best_index_])
```

チューニングの結果、最も精度の高かったモデルを用います。学習用データ全体でモデルの学習をして、テストデータに対して予測を行います。
```py
# 学習データ全体でモデルの学習をする
tune_rfr.best_estimator_.fit(train_x, train_y)

# テストデータに対して予測する
predict = tune_rfr.best_estimator_.predict(test_x)
```

最後に提出ファイルを作成します。
```py
# 提出用ファイルの作成
submit_1 = pd.DataFrame({'Id': test['Id'], 'SalePrice': predict})
submit_1.to_csv('submission.csv', index=False)
```

# 最後に
今回は各変数について細かい分析を行っていないので、もう少し深く分析をする記事も書きたいと思ってます！