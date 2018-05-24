## First Part: Load the data:
import pandas as pd
import numpy as np


def shuffle(x_, y_):
    s = np.arange(x_.shape[0])
    s = np.random.shuffle(s)

    x_re = x_[s]
    y_re = y_[s]

    x_re = np.reshape(x_re, (len(x_), 48, 48))
    y_re = np.reshape(y_re, (len(y_)))
    return x_re, y_re


def read_fer(path):
    # train_path = "C:\\Users\\Jiaming Nie\\Documents\\Work-DeepGlint\Facial\datasets\\train.csv"
    data = pd.read_csv(path, dtype='a')
    label = np.array(data['emotion'])
    img_data = np.array(data['pixels'])

    N_sample = label.size

    x_data = np.zeros((N_sample, 48 * 48))
    # train_label = np.zeros((N_sample, 7), dtype=int)
    y_label = np.zeros(N_sample, dtype=int)
    # print(train_label)

    for i in range(N_sample):
        x = img_data[i]
        x = np.fromstring(x, dtype=float, sep=' ')
        x_max = x.max()
        x = x / (x_max + 0.0001)
        # print x_max
        # print x
        x_data[i] = x
        y_label[i] = int(label[i])
        # train_label[i, label[i]] = 1 #This step seems direct one-hot encoding
        # print(y_label[i])
        #    img_x = np.reshape(x, (48, 48))
        #    plt.subplot(10,10,i+1)
        #    plt.axis('off')
        #    plt.imshow(img_x, plt.cm.gray)

    x_data = np.reshape(x_data, (len(x_data), 48, 48))
    return x_data, y_label


def ReadData_fer():
    # ubuntu path
    # path_train = "/home/jiaming/code/DeepGlint-Work/Facial/datasets/train.csv"
    # path_test = "/home/jiaming/code/DeepGlint-Work/Facial/datasets/test.csv"

    # windows path
    path_train = "/train/trainset/1/train.csv"
    path_test = "/train/trainset/1/test.csv"
    path_vali = "/train/trainset/1/val.csv"

    x_train, y_train = read_fer(path_train)
    x_test, y_test = read_fer(path_test)
    x_vali, y_vali = read_fer(path_vali)

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    x_vali, y_vali = shuffle(x_vali, y_vali)

    return x_train, y_train, x_test, y_test, x_vali, y_vali


def zca_whitening(X):
    """
        Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
        INPUT:  X: [M x N] matrix.
            Rows: Variables
            Columns: Observations
        OUTPUT: ZCAMatrix: [M x M] matrix
        """
    mean_ = np.mean(X)
    X = X - mean_
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True)  # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    # U: [M x M] eigenvectors of sigma.
    # S: [M x 1] eigenvalues of sigma.
    # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 0.1
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))  # [M x M]
    return ZCAMatrix


def normalization(x_):
    length = len(x_)

    for i in range(length):
        x_[i] = zca_whitening(x_[i])

    return x_


def oneHot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

x_train,y_train,x_test,y_test,x_vali,y_vali = ReadData_fer()

# Normalization
#x_train = normalization(x_train)
#x_test = normalization(x_test)
#x_vali = normalization(x_vali)

x_train = x_train.reshape((len(x_train), 48, 48, 1))
x_test = x_test.reshape((len(x_test), 48, 48, 1))
x_vali = x_vali.reshape((len(x_vali),48,48,1))

print(x_train.shape)
print(x_test.shape)

from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D
from keras.layers import add, Flatten
#from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import to_categorical
import os
from keras.optimizers import *
import numpy as np
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))

batch_size=256
EPOCH=60
NB_CLASS=7
IM_WIDTH=48
IM_HEIGHT=48
CHANNEL = 1

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same',kernel_initializer= 'glorot_uniform',kernel_regularizer=l2(1e-5))
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same',kernel_initializer= 'glorot_uniform',kernel_regularizer=l2(1e-5))
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def bottleneck_Block(inpt,nb_filters,strides=(1,1),with_conv_shortcut=False):
    k1,k2,k3=nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same',kernel_initializer= 'glorot_uniform',kernel_regularizer=l2(1e-5))
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same',kernel_initializer= 'glorot_uniform',kernel_regularizer=l2(1e-5))
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same',kernel_initializer= 'glorot_uniform',kernel_regularizer=l2(1e-5))
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def resnet_34(width,height,channel,classes):
    inpt = Input(shape=(width, height, channel))
    x = ZeroPadding2D((3, 3))(inpt)

    #conv1
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))

    #conv3_x
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(6,6))(x)
    x = Dropout(0.5)(x)
    #conv4_x
    #x = identity_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    #x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    #x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    #x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    #x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    #x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))

    #conv5_x
    #x = identity_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    #x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    #x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    #x = AveragePooling2D(pool_size=(7, 7))(x)
    #x = AveragePooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model


def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def check_print():
    # Create a Keras Model
    model = resnet_34(IM_WIDTH,IM_HEIGHT,1,NB_CLASS)
    #model.summary()
    # Save a PNG of the Model Build
    #plot_model(model, to_file='resnet.png')
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc',top_k_categorical_accuracy])
    print('Model Compiled')
    return model

if __name__ == '__main__':

    x_train, y_train, x_test, y_test, x_vali, y_vali = ReadData_fer()

    x_train = np.reshape(x_train,(len(x_train),48,48,1))
    x_test = np.reshape(x_test,(len(x_test),48,48,1))
    x_vali = np.reshape(x_vali,(len(x_vali),48,48,1))

    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_test.shape)
    #print(y_test.shape)
    #print(x_vali.shape)
    #print(y_vali.shape)

    model = check_print()
    model.summary()
    score = model.evaluate(x_train,to_categorical(y_train),verbose=0)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])

    model.fit(x_train,to_categorical(y_train),
              epochs=EPOCH,
              verbose=1,
              validation_data=(x_vali, to_categorical(y_vali))
    )
    # model.fit_generator(train_generator,validation_data=vaild_generator,epochs=EPOCH,steps_per_epoch=train_generator.n/batch_size
    #                     ,validation_steps=vaild_generator.n/batch_size)
    # model.save('resnet_50.h5')
    # loss,acc,top_acc=model.evaluate_generator(test_generator, steps=test_generator.n / batch_size)
    # print('Test result:loss:%f,acc:%f,top_acc:%f' % (loss, acc, top_acc))
