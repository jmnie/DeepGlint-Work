import numpy as np
import pandas as pd
import scipy.io as sio
#import torch
#import torch.utils.data as data_utils

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

    x_data = np.reshape(x_data,(len(x_data),48,48))
    return x_data, y_label

def ReadData_fer():
    # ubuntu path
    #path_train = "/home/jiaming/code/DeepGlint-Work/Facial/datasets/train.csv"
    #path_test = "/home/jiaming/code/DeepGlint-Work/Facial/datasets/test.csv"

    # windows path
    path_train = "C:\\Users\Jiaming Nie\Documents\GitHub\DeepGlint-Work\Facial\datasets\\train.csv"
    path_test = "C:\\Users\Jiaming Nie\Documents\GitHub\DeepGlint-Work\Facial\datasets\\test.csv"
    path_vali = "C:\\Users\Jiaming Nie\Documents\GitHub\DeepGlint-Work\Facial\datasets\\val.csv"
    x_train, y_train = read_fer(path_train)
    x_test, y_test = read_fer(path_test)
    x_vali, y_vali = read_fer(path_vali)

    #x_train, y_train = shuffle(x_train, y_train)
    #x_test, y_test = shuffle(x_test, y_test)
    #x_vali, y_vali = shuffle(x_vali, y_vali)

    return x_train,y_train,x_test,y_test,x_vali,y_vali

def shuffle(x_,y_):
    s = np.arange(x_.shape[0])
    s = np.random.shuffle(s)

    x_re = x_[s]
    y_re = y_[s]

    x_re = np.reshape(x_re,(len(x_),48,48))
    y_re = np.reshape(y_re,(len(y_)))
    return x_re,y_re


# def fer_read_tensor():
#     # ubuntu path
#     path_train = "/home/jiaming/code/DeepGlint-Work/Facial/datasets/train.csv"
#     path_test = "/home/jiaming/code/DeepGlint-Work/Facial/datasets/test.csv"
#
#     x_train, y_train, x_test, y_test = ReadData_fer(path_train,path_test)
#     train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
#     train_loader = data_utils.DataLoader(tr   ain, batch_size=50, shuffle=True)

#
# test = np.random.rand(2,2)
# print(test,np.mean(test),np.std(test))
# mean = np.mean(test)
# std = np.std(test)
# test -= mean
# print(test)
# test /= std
# print(test)

def save_to_mat():
    x_train, y_train, x_test, y_test, x_vali, y_vali = ReadData_fer()
    x_train = x_train.reshape((len(x_train), 48, 48, 1))
    x_test = x_test.reshape((len(x_test), 48, 48, 1))
    x_vali = x_vali.reshape((len(x_vali), 48, 48, 1))

    train_name = 'x_train.mat'
    vali_name = 'x_vali.mat'
    test_name = 'x_test.mat'
    sio.savemat(train_name, {'x_train': x_train})
    sio.savemat(test_name, {'x_test': x_test})
    sio.savemat(vali_name, {'x_vali': x_vali})
    np.savetxt('y_train.txt', y_train)
    np.savetxt('y_test.txt', y_test)
    np.savetxt('y_vali.txt', y_vali)

save_to_mat()