---
title: "titanic digit recognizerに挑戦!"
emoji: "👋"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["python", "機械学習", "初心者", "プログラミング"]
published: true
---
# 初めに
この記事では、kaggleのDigit Recognizerコンペについて説明していきます！0から9までの手書き文字を判別するコンペです。なお、コードは以下を参考にして下さい。
https://www.kaggle.com/code/iketuba/digit-recognizer-2022-06-25

# ライブラリのインポート
以下のライブラリをインポートします。
```py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
```

# データの準備
データを読み込みます。
```py
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
sample = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
```

データを画像とラベルに分けます。
```py
train_x = train.drop(['label'], axis=1)
train_y = train['label']
test_x = test.copy()
```

データのサイズを確認します。
```py
train_x.shape, train_y.shape, test_x.shape
```

学習用の画像が42000枚、テストの画像が28000枚あることが分かります。また、画像データが2次元になっていないため、後ほどサイズを変更します。<br>次に欠損値を確認します。
```py
train_x.isnull().any().describe()
test_x.isnull().any().describe()
```

出力から欠損値はないことが分かります。<br>また、pixelの値は0~255なので、255で割ることで正規化します。
```py
train_x = train_x / 255.0
test_x = test_x / 255.0
```

次に、データの形状を2次元に変更します。pixel数が784個あるため28×28に変換します。
```py
train_x = train_x.values.reshape(-1, 28, 28, 1)
test_x = test_x.values.reshape(-1, 28, 28, 1)
```

続いて、画像のラベルをワンホットエンコーディングします。
```py
train_y = to_categorical(train_y, num_classes=10)
```

学習用データを訓練データと検証データに分割します。
```py
tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
```

# モデルの定義
kerasを用いてモデルを定義します。
```py
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```

モデルをコンパイルします。最適化の手法にはAdamを用います。
```py
model.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=["accuracy"])
```

エポック数とバッチサイズを定義します。
```py
epochs = 2
batch_size = 64
```

モデルの学習を行います。
```py
history = model.fit(tr_x, tr_y, 
                    batch_size=batch_size, epochs=epochs, 
                    validation_data=(va_x, va_y))
```

# 予測と提出
学習したモデルを使って、テストデータに対する予測値を出力します。
```py
results = model.predict(test_x)
results = np.argmax(results, axis=1)
results = pd.Series(results)
```

最後にsample_submission.csvの形に整えます。
```py
submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': results})
submission.to_csv('submission.csv', index=False)
```

# 最後に
最後まで読んでいただいてありがとうございました～