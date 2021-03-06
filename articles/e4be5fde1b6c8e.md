---
title: "kaggle digit recognizerã«ææ¦!"
emoji: "ð"
type: "tech" # tech: æè¡è¨äº / idea: ã¢ã¤ãã¢
topics: ["python", "æ©æ¢°å­¦ç¿", "åå¿è", "ãã­ã°ã©ãã³ã°"]
published: true
---
# åãã«
ãã®è¨äºã§ã¯ãkaggleã®Digit Recognizerã³ã³ãã«ã¤ãã¦èª¬æãã¦ããã¾ãï¼0ãã9ã¾ã§ã®ææ¸ãæå­ãå¤å¥ããã³ã³ãã§ãããªããã³ã¼ãã¯ä»¥ä¸ãåèã«ãã¦ä¸ããã
https://www.kaggle.com/code/iketuba/digit-recognizer-2022-06-25

# ã©ã¤ãã©ãªã®ã¤ã³ãã¼ã
ä»¥ä¸ã®ã©ã¤ãã©ãªãã¤ã³ãã¼ããã¾ãã
```py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
```

# ãã¼ã¿ã®æºå
ãã¼ã¿ãèª­ã¿è¾¼ã¿ã¾ãã
```py
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
sample = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
```

ãã¼ã¿ãç»åã¨ã©ãã«ã«åãã¾ãã
```py
train_x = train.drop(['label'], axis=1)
train_y = train['label']
test_x = test.copy()
```

ãã¼ã¿ã®ãµã¤ãºãç¢ºèªãã¾ãã
```py
train_x.shape, train_y.shape, test_x.shape
```

å­¦ç¿ç¨ã®ç»åã42000æããã¹ãã®ç»åã28000æãããã¨ãåããã¾ããã¾ããç»åãã¼ã¿ã2æ¬¡åã«ãªã£ã¦ããªããããå¾ã»ã©ãµã¤ãºãå¤æ´ãã¾ãã<br>æ¬¡ã«æ¬ æå¤ãç¢ºèªãã¾ãã
```py
train_x.isnull().any().describe()
test_x.isnull().any().describe()
```

åºåããæ¬ æå¤ã¯ãªããã¨ãåããã¾ãã<br>ã¾ããpixelã®å¤ã¯0~255ãªã®ã§ã255ã§å²ããã¨ã§æ­£è¦åãã¾ãã
```py
train_x = train_x / 255.0
test_x = test_x / 255.0
```

æ¬¡ã«ããã¼ã¿ã®å½¢ç¶ã2æ¬¡åã«å¤æ´ãã¾ããpixelæ°ã784åãããã28Ã28ã«å¤æãã¾ãã
```py
train_x = train_x.values.reshape(-1, 28, 28, 1)
test_x = test_x.values.reshape(-1, 28, 28, 1)
```

ç¶ãã¦ãç»åã®ã©ãã«ãã¯ã³ãããã¨ã³ã³ã¼ãã£ã³ã°ãã¾ãã
```py
train_y = to_categorical(train_y, num_classes=10)
```

å­¦ç¿ç¨ãã¼ã¿ãè¨ç·´ãã¼ã¿ã¨æ¤è¨¼ãã¼ã¿ã«åå²ãã¾ãã
```py
tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
```

# ã¢ãã«ã®å®ç¾©
kerasãç¨ãã¦ã¢ãã«ãå®ç¾©ãã¾ãã
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

ã¢ãã«ãã³ã³ãã¤ã«ãã¾ããæé©åã®ææ³ã«ã¯Adamãç¨ãã¾ãã
```py
model.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=["accuracy"])
```

ã¨ããã¯æ°ã¨ããããµã¤ãºãå®ç¾©ãã¾ãã
```py
epochs = 2
batch_size = 64
```

ã¢ãã«ã®å­¦ç¿ãè¡ãã¾ãã
```py
history = model.fit(tr_x, tr_y, 
                    batch_size=batch_size, epochs=epochs, 
                    validation_data=(va_x, va_y))
```

# äºæ¸¬ã¨æåº
å­¦ç¿ããã¢ãã«ãä½¿ã£ã¦ããã¹ããã¼ã¿ã«å¯¾ããäºæ¸¬å¤ãåºåãã¾ãã
```py
results = model.predict(test_x)
results = np.argmax(results, axis=1)
results = pd.Series(results)
```

æå¾ã«sample_submission.csvã®å½¢ã«æ´ãã¾ãã
```py
submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': results})
submission.to_csv('submission.csv', index=False)
```

# æå¾ã«
æå¾ã¾ã§èª­ãã§ããã ãã¦ãããã¨ããããã¾ããï½