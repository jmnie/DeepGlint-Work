from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
from mxnet.gluon.model_zoo import vision as models
from mxnet import init

def normalization(data):
    data = data * (1. / 255) - 0.5
    return data

mx.random.seed(1)

ctx = mx.cpu()
batch_size = 1
channel = 3
size = 224
epoch = 5

path_out = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/Processed/mxnet_list/'
data_dir = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/aligned_final/'

train_iter = mx.image.ImageIter(batch_size = batch_size, data_shape=(channel, size, size), label_width=1,
                                   path_imglist=path_out+'test.lst',path_root=data_dir)

test_iter = mx.image.ImageIter(batch_size = batch_size, data_shape=(channel, size, size), label_width=1,
                                path_imglist=path_out+'vali.lst',path_root=data_dir)

alex_net = gluon.nn.Sequential()

with alex_net.name_scope():
    #  First convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=96, kernel_size=11, strides=(4,4), activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))    
    #  Second convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=192, kernel_size=5, activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=(2,2)))            
    # Third convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
    # Fourth convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu')) 
    # Fifth convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))    
    # Flatten and apply fullly connected layers
    alex_net.add(gluon.nn.Flatten())
    alex_net.add(gluon.nn.Dense(4096, activation="relu"))
    alex_net.add(gluon.nn.Dense(4096, activation="relu"))
    alex_net.add(gluon.nn.Flatten())
    alex_net.add(gluon.nn.Dense(8))

alex_net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
trainer = gluon.Trainer(alex_net.collect_params(), 'sgd', {'learning_rate': .1})
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# pretrained_net = models.resnet18_v2(pretrained=True)
# finetune_net = models.resnet18_v2(classes=8)
# finetune_net.features = pretrained_net.features
# finetune_net.output.initialize(init.Xavier())



def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()    
    #acc = 0
    for data_ in data_iterator:
        image = normalization(data_.data[0]).as_in_context(ctx)
        label = normalization(data_.label[0]).as_in_context(ctx)
        output = net(image)
        predictions = nd.argmax(output, axis=1)
        #print("Label ",label)
        #print("Predict ",predictions)
        acc.update(preds=predictions,labels=label)
        #print("Accuracy :",acc.get()[1])

    #print("Overall Acc:",temp_acc/i)
    return acc.get()[1]

# def normalization(data):
#     data = data * (1. / 255) - 0.5
#     return data

epochs = 3
smoothing_constant = .01

moving_loss  = 0
print(alex_net)

train_accuracy = evaluate_accuracy(train_iter,alex_net)
print("Train Accuracy",train_accuracy)

for e in range(epochs):

    train_loss = 0.
    train_acc = 0.

    i = 0
    for data_ in train_iter:
        data = normalization(data_.data[0])
        label = normalization(data_.label[0])

        with autograd.record():
            output = alex_net(data)
            loss = softmax_cross_entropy(output, label)
        
        for l in loss:
            l.backward()
        #loss.backward()
        trainer.step(batch_size)

        curr_loss = nd.mean(loss).asscalar()
        print("curr_loss",curr_loss)
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
        i = i + 1   
        print("i :",i)
        print("Loss :",moving_loss)
        #train_accuracy += evaluate_accuracy(tr)
        #print("Output ",output)
        #print("Label ",label)
        
    train_acc = evaluate_accuracy(train_iter,alex_net)
    test_acc = evaluate_accuracy(test_iter, alex_net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_acc, test_acc))    