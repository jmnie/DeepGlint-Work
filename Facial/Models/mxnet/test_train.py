from resnet_mx import *
import mxnet as mx
from mxnet import gluon
from mxnet import nd, autograd
import numpy as np
import scipy.io


def compile(model):
    ctx = mx.cpu()
    init = mx.init.Xavier(factor_type="in", magnitude=255)
    print(init.dumps())
    model.collect_params().initialize(init, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': .001})
    return model, trainer

def getLoss():
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    return softmax_cross_entropy

def evaluate_accuracy(data,label, net):
    ctx = mx.cpu()
    acc = mx.metric.Accuracy()

    for i in range(len(label)):
        data_ = data[i]
        label_ = label[i]
        output = net(data)
        predeictions = nd.argmax(output,axis=1)
        acc.update(preds=predictions, labels=label)

    return acc.get()[1]

if __name__ == '__main__':

    dataset_path = '/media/jiaming/Seagate Backup Plus Drive/fer2013/'
    x_train_path = dataset_path + 'x_train.mat'
    y_train_path = dataset_path + 'y_train.txt'
    x_vali_path = dataset_path + 'x_vali.mat'
    y_vali_path = dataset_path + 'y_vali.txt'
    
    x_train = nd.array(scipy.io.loadmat(x_train_path)['x_train'])
    print(x_train.shape)
    y_train = nd.array(np.loadtxt(y_train_path))
    x_vali = nd.array(scipy.io.loadmat(x_vali_path)['x_vali'])
    print(x_vali.shape)
    y_vali = nd.array(np.loadtxt(y_vali_path))


    epochs = 5
    smoothing_constant = .01
    batch_size = 1

    model = resnet18_v2()
    model,trainer = compile(model)

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    model_para = params = {'pretrained':True,
                            }

    for e in range(epochs):
        for i in range(len(x_train)):
            data = x_train[i]
            label = y_train[i]
            with autograd.record():
                output = model(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            
            ##########################
            #  Keep a moving average of the losses
            ##########################
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                        else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
                
        test_accuracy = evaluate_accuracy(x_vali, model)
        train_accuracy = evaluate_accuracy(x_train, model)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))



    #model = resnet18_v2()
    #compile(model)

