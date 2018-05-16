import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

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

    x_data = np.reshape(x_data,(len(x_data),1,48,48))
    return x_data, y_label

def ReadData_fer():
    # ubuntu path
    path_train = "/home/jiaming/code/DeepGlint-Work/Facial/datasets/train.csv"
    path_test = "/home/jiaming/code/DeepGlint-Work/Facial/datasets/test.csv"

    # windows path
    #path_train =
    #path_test =
    x_train, y_train = read_fer(path_train)
    x_test, y_test = read_fer(path_test)

    return x_train,y_train,x_test,y_test

# def fer_read_tensor():
#     # ubuntu path
#     path_train = "/home/jiaming/code/DeepGlint-Work/Facial/datasets/train.csv"
#     path_test = "/home/jiaming/code/DeepGlint-Work/Facial/datasets/test.csv"
#
#     x_train, y_train, x_test, y_test = ReadData_fer(path_train,path_test)
#     train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
#     train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)




