from keras.datasets import mnist
from keras.layers import Conv2D, Flatten, Dense, Activation, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils

lr = 1e-4

(x_train, x_label), (y_train, y_label) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
y_train = y_train.reshape(-1, 28, 28, 1)
x_label = np_utils.to_categorical(x_label, num_classes=10)
y_label = np_utils.to_categorical(y_label, num_classes=10)

model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(strides=(1, 1), padding='same'))
model.add(Dropout(rate=0.5))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(1, 1), padding='same'))
model.add(Dropout(rate=0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=lr)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, x_label, epochs=2, batch_size=32)
print(model.evaluate(y_train, y_label))
