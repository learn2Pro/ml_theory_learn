from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import metrics
from keras import utils
import numpy as np

x_train = np.random.random((10000, 128))
y_train = utils.to_categorical(np.random.randint(10, size=(10000, 1)), num_classes=10)
x_test = np.random.random((1000, 128))
y_test = utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

# Keras中主要的模型是Sequential模型，Sequential是一系列网络层按顺序构成的栈。
model = Sequential()
# 将一些网络层通过.add()堆叠起来，就构成了一个模型
model.add(Dense(64, input_dim=128))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))
# 完成模型的搭建后，我们需要使用.compile()方法来编译模型：
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
model.fit(x_train, y_train, batch_size=256,epochs=10)
print(model.evaluate(x_test,y_test,batch_size=256))
