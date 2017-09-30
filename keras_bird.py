from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.optimizers import Adam

import game.wrapped_flappy_bird as game
from keras.models import Sequential

input_row, input_column = 80, 80
channel = 3
batch_size = 4
lr = 1e-4


#####
######    (80*80)
##          (40*40)
def buildModle():
    model = Sequential()
    # filters：卷积核的数目（即输出的维度）
    # kernal window(8*8)
    # filters = 64
    model.add(Conv2D(64, kernel_size=(8, 8), activation='relu', input_shape=(input_row, input_column, channel)))
    model.add(MaxPooling2D(strides=(1, 1)))
    model.add(Dropout(rate=0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    return model
