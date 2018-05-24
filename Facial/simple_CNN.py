import csv 
import pandas as pd 
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import time 
from sklearn import metrics
from Models.LoadData import ReadData_fer

## Read Train Data
def ReadData(path):
    #train_path = "C:\\Users\\Jiaming Nie\\Documents\\Work-DeepGlint\Facial\datasets\\train.csv"
    data = pd.read_csv(path, dtype = 'a')
    label = np.array(data['emotion'])
    img_data = np.array(data['pixels'])
    
    N_sample = label.size

    x_data = np.zeros((N_sample, 48*48))
    #train_label = np.zeros((N_sample, 7), dtype=int)
    y_label = np.zeros(N_sample,dtype = int)
    #print(train_label)

    for i in range(N_sample):
        x = img_data[i]
        x = np.fromstring(x, dtype=float, sep=' ')
        x_max = x.max()
        x = x/(x_max+0.0001)
        # print x_max
        # print x
        x_data[i] = x
        y_label[i] = int(label[i])
        #train_label[i, label[i]] = 1 #This step seems direct one-hot encoding
        #print(y_label[i])
        #    img_x = np.reshape(x, (48, 48))
        #    plt.subplot(10,10,i+1)
        #    plt.axis('off')
        #    plt.imshow(img_x, plt.cm.gray)

    return x_data, y_label


def oneHot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

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
#train_path = "C:\\Users\\Jiaming Nie\\Documents\\Work-DeepGlint\Facial\datasets\\train.csv"
#test_path = "C:\\Users\Jiaming Nie\Documents\Work-DeepGlint\Facial\datasets\\test.csv"
#x_train, y_train = ReadData(train_path)
#x_test, y_test = ReadData(test_path)
#x_train = x_train.reshape(len(x_train), 48, 48, 1)
#x_test = x_test.reshape(len(x_test), 48, 48, 1)

x_train,y_train,x_test,y_test,x_vali,y_vali = ReadData_fer()

# # Normalization
x_train = normalization(x_train)
x_test = normalization(x_test)
x_vali = normalization(x_vali)

x_train = x_train.reshape((len(x_train), 48, 48, 1))
x_test = x_test.reshape((len(x_test), 48, 48, 1))
x_vali = x_vali.reshape((len(x_vali),48,48,1))


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train)
print(y_test)
train_num = y_train.size
test_num = y_test.size

LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D,Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import *
from keras.layers import Dense
batch_size=30
epochs=20
num_class = 7
num_output = 7
# Initialising the CNN
model = Sequential()

model = Sequential()
#model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',#input_shape=(48,48,1)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(48, 48, 1)))

model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
#model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
#model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
#model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
#model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
#model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
#model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
#model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
#model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))



# Compiling the CNN
#sgd= SGD(lr=0.01,momentum=0.99, decay=0.9999, nesterov=True)

adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer = adam , loss = 'categorical_crossentropy', metrics = ['accuracy'])

# # Step 1 - Convolution
# #classifier.add(Convolution2D(32, 3, 3, border_mode='same',input_shape = (48, 48, 1), activation = 'relu'))
# #classifier.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(48, 48, 1), activation='relu'))
# classifier.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation="relu", padding="same"))
# # Step 2 - Pooling
# classifier.add(MaxPooling2D(pool_size = (5, 5)))
# #classifier.add(Dropout(0.25))
# # Adding a second convolutional layer
# classifier.add(Conv2D(32, (3, 3), activation="relu"))
# #classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (5, 5)))
# #classifier.add(Dropout(0.25))
# # Step 3 - Flattening
# classifier.add(Flatten())
#
# # Step 4 - Full connection
# classifier.add(Dense(activation="relu", units=500)) # the output_dim is chosen by experience
# classifier.add(Dense(activation="softmax", units=7))
#
# # Compiling the CNN
# sgd= SGD(lr=10e-5,momentum=0.99, decay=0.9999, nesterov=True)
# classifier.compile(optimizer =sgd , loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
# score = classifier.evaluate(x_train, oneHot(y_train), verbose=0)
# print('Train loss:', score[0])
# print('Train accuracy:', score[1])
#
# score_vali = classifier.evaluate(x_vali, oneHot(y_vali),verbose=0)
# print('Vali loss',score_vali[0])
# print('Vali Accuracy',score_vali[1])


# results = []
# start_time = time.time()
# history = classifier.fit(x_train, oneHot(y_train),
#            batch_size=batch_size,
#            epochs=epochs,
#            verbose=1,
#            validation_data=(x_vali, oneHot(y_vali)))
#
# average_time_per_epoch = (time.time() - start_time) / epochs
# results.append((history, average_time_per_epoch))
# plt.style.use('ggplot')
# ax1 = plt.subplot2grid((2, 2), (0, 0))
# ax1.set_title('Accuracy')
# ax1.set_ylabel('Accuracy')
# ax1.set_xlabel('Epochs')
# ax2 = plt.subplot2grid((2, 2), (1, 0))
# ax2.set_title('Loss')
# ax2.set_ylabel('Loss')
# ax2.set_xlabel('Epochs')
# ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
# ax3.set_title('Time')
# ax3.set_ylabel('Seconds')
#
# for result in results:
#     ax1.plot(result[0].epoch, result[0].history['val_acc'],label='Test')
#     ax1.plot(result[0].epoch, result[0].history['acc'],label = 'Train')
#     ax2.plot(result[0].epoch, result[0].history['val_loss'],label = 'Test')
#     ax2.plot(result[0].epoch, result[0].history['loss'],label = 'Train')
#
# ax1.legend()
# ax2.legend()
# ax3.bar(np.arange(len(results)), [x[1] for x in results],
#          align='center')
# plt.tight_layout()
# plt.show()
#
# score = classifier.evaluate(x_test, oneHot(y_test), verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# # Results
#
# #score = model.evaluate(x_test, one_hot(y_test), verbose=0)
#
# predictions_one_hot = classifier.predict(x_test)
# predictions = predictions_one_hot.argmax(1)

# print("Testing Accuracy: {}%".format(100*accuracy))

# print("")
# print("Accuracy: {}%".format(100*metrics.accuracy_score(y_test, predictions)))
# print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
# print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))
#
# print("")
# print("Confusion Matrix:")
# confusion_matrix = metrics.confusion_matrix(y_test, predictions)
# # confusion_matrix = metrics.confusion_matrix(one_hot(y_test), predictions_one_hot)
# print(confusion_matrix)
# normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
#
# print("")
# print("Confusion matrix (normalised to % of total test data):")
# print(normalised_confusion_matrix)
# print("Note: training and testing data is not equally distributed amongst classes, ")
# print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")
#
# # Plot Results:
# width = 6
# height = 6
# plt.figure(figsize=(width, height))
# plt.imshow(
#     normalised_confusion_matrix,
#     interpolation='nearest',
#     cmap=plt.cm.rainbow
# )
# plt.title("Confusion matrix \n(normalised to % of total test data)")
# plt.colorbar()
# tick_marks = np.arange(num_output)
# plt.xticks(tick_marks, LABELS, rotation=90)
# plt.yticks(tick_marks, LABELS)
# plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()