# coding=utf-8
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D
from keras.layers import add, Flatten
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import to_categorical
import os
import numpy as np
import matplotlib.pyplot as plt
from LoadData import ReadData_fer
import time

# Global Constants
NB_CLASS=7
IM_WIDTH=48
IM_HEIGHT=48
CHANNEL = 1

x_train, y_train, x_test, y_test, x_vali, y_vali = ReadData_fer()

x_train = np.reshape(x_train,(len(x_train),48,48,1))
x_test = np.reshape(x_test,(len(x_test),48,48,1))
x_vali = np.reshape(x_vali,(len(x_vali),48,48,1))
# train_root='/home/faith/keras/dataset/traindata/'
# vaildation_root='/home/faith/keras/dataset/vaildationdata/'
# test_root='/home/faith/keras/dataset/testdata/'

def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

batch_size=32
EPOCH=60

# # train data
# train_datagen = ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     rescale=1./255
# )
# train_generator = train_datagen.flow_from_directory(
#     train_root,
#     target_size=(IM_WIDTH, IM_HEIGHT),
#     batch_size=batch_size,
#     shuffle=True
# )
#
# # vaild data
# vaild_datagen = ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     rescale=1./255
# )
# vaild_generator = train_datagen.flow_from_directory(
#     vaildation_root,
#     target_size=(IM_WIDTH, IM_HEIGHT),
#     batch_size=batch_size,
# )
#
# # test data
# test_datagen = ImageDataGenerator(
#     rescale=1./255
# )
# test_generator = train_datagen.flow_from_directory(
#     test_root,
#     target_size=(IM_WIDTH, IM_HEIGHT),
#     batch_size=batch_size,
# )

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
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def bottleneck_Block(inpt,nb_filters,strides=(1,1),with_conv_shortcut=False):
    k1,k2,k3=nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
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

    #conv4_x
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))

    #conv5_x
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    #x = AveragePooling2D(pool_size=(7, 7))(x)
    x = AveragePooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model

def resnet_50(width,height,channel,classes):
    inpt = Input(shape=(width, height, channel))
    #print("inpt shape",inpt.shape)

    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = bottleneck_Block(x, nb_filters=[64,64,256],strides=(1,1),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64,64,256])
    x = bottleneck_Block(x, nb_filters=[64,64,256])

    #conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])

    #conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])

    #conv5_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model

def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def check_print():
    # Create a Keras Model
    model = resnet_34(IM_WIDTH,IM_HEIGHT,1,NB_CLASS)
    model.summary()
    # Save a PNG of the Model Build
    #plot_model(model, to_file='resnet.png')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',top_k_categorical_accuracy])
    print('Model Compiled')
    return model


if __name__ == '__main__':

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(x_vali.shape)
    print(y_vali.shape)
    #inpt = Input(shape=(48, 48, 1))
    #print(inpt.shape)

    if os.path.exists('resnet_34.h5'):
        model=load_model('resnet_34.h5')
    else:
        model=check_print()

    results = []
    start_time = time.time()

    history = model.fit(x_train, to_categorical(y_train),
              batch_size = batch_size,
              epochs= EPOCH,
              validation_data=(x_vali,to_categorical(y_vali)))

    #scores = model.evaluate(x_test,to_categorical(y_test))
    #print("Accuracy: ",scores[1])
    #print("Scores[0] :",scores[0])

    average_time_per_epoch = (time.time() - start_time) / EPOCH
    results.append((history, average_time_per_epoch))
    plt.style.use('ggplot')
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.set_title('Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epochs')
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.set_title('Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax3.set_title('Time')
    ax3.set_ylabel('Seconds')

    for result in results:
        ax1.plot(result[0].epoch, result[0].history['val_acc'], label='Test')
        ax1.plot(result[0].epoch, result[0].history['acc'], label='Train')
        ax2.plot(result[0].epoch, result[0].history['val_loss'], label='Test')
        ax2.plot(result[0].epoch, result[0].history['loss'], label='Train')

    ax1.legend()
    ax2.legend()
    ax3.bar(np.arange(len(results)), [x[1] for x in results],
            align='center')
    plt.tight_layout()
    plt.show()

    score = model.evaluate(x_test, to_categorical(y_test), verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # model.fit_generator(train_generator,validation_data=vaild_generator,epochs=EPOCH,steps_per_epoch=train_generator.n/batch_size
    #                     ,validation_steps=vaild_generator.n/batch_size)
    # model.save('resnet_50.h5')
    # loss,acc,top_acc=model.evaluate_generator(test_generator, steps=test_generator.n / batch_size)
    # print('Test result:loss:%f,acc:%f,top_acc:%f' % (loss, acc, top_acc))