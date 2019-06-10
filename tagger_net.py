from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GRU, Dropout, Permute, Reshape
from keras.optimizers import SGD

def MusicTaggerCNN():
    
    CNNmodel = Sequential()
    CNNmodel.add(Conv2D(64, kernel_size=3, activation='relu', padding ='same', input_shape=(1,96,1366)))
    CNNmodel.add(MaxPool2D(pool_size=(2,4)))
    CNNmodel.add(Dropout(0.1))
    CNNmodel.add(Conv2D(128, kernel_size=3, activation='relu', padding ='same'))
    CNNmodel.add(MaxPool2D(pool_size=(2,4)))
    CNNmodel.add(Dropout(0.1))
    CNNmodel.add(Conv2D(128, kernel_size=3, activation='relu', padding ='same'))
    CNNmodel.add(MaxPool2D(pool_size=(2,4)))
    CNNmodel.add(Dropout(0.1))
    CNNmodel.add(Conv2D(128, kernel_size=3, activation='relu', padding ='same'))
    CNNmodel.add(MaxPool2D(pool_size=(3,5)))
    CNNmodel.add(Dropout(0.1))
    CNNmodel.add(Conv2D(64, kernel_size=3, activation='relu', padding ='same'))
    CNNmodel.add(MaxPool2D(pool_size=(4,4)))
    CNNmodel.add(Dropout(0.1))
    CNNmodel.add(Flatten())
    CNNmodel.add(Dense(10, activation='softmax'))
    CNNmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return CNNmodel

def MusicTaggerCRNN():
    CRNNmodel = Sequential()
    CRNNmodel.add(Conv2D(64, kernel_size=3, activation='relu', padding ='same', input_shape=(1,96,1366)))
    CRNNmodel.add(MaxPool2D(pool_size=(2,2)))
    #CRNNmodel.add(Dropout(0.1))
    CRNNmodel.add(Conv2D(128, kernel_size=3, activation='relu', padding ='same'))
    CRNNmodel.add(MaxPool2D(pool_size=(3,3)))
    #CRNNmodel.add(Dropout(0.1))
    CRNNmodel.add(Conv2D(128, kernel_size=3, activation='relu', padding ='same'))
    CRNNmodel.add(MaxPool2D(pool_size=(4,4)))
    #CRNNmodel.add(Dropout(0.1))
    CRNNmodel.add(Conv2D(128, kernel_size=3, activation='relu', padding ='same'))
    CRNNmodel.add(MaxPool2D(pool_size=(4,4)))
    #CRNNmodel.add(Dropout(0.1))   
    CRNNmodel.add(Permute((3, 1, 2)))
    CRNNmodel.add(Reshape((14, 128)))
    CRNNmodel.add(GRU(32, return_sequences=True))
    CRNNmodel.add(GRU(32, return_sequences=False))
    #CRNNmodel.add(Dropout(0.3))
    CRNNmodel.add(Dense(10, activation='softmax'))
    #opt = SGD(lr=0.05)
    CRNNmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return CRNNmodel