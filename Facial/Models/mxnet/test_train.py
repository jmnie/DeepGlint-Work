import mxnet as mx
from mxnet import gluon
from mxnet import nd, autograd
from mxnet.gluon.model_zoo import vision
import numpy as np
import scipy.io
from time import time

def evaluate_accuracy(data_iterator, net, ctx):
    acc = mx.metric.Accuracy()

    data = data_iterator.data[0]
    label = data_iterator.label[0]
    
    for i in range(len(data)):
        temp_data = data[i].as_in_context(ctx)
        temp_label = label[i].as_in_context(ctx)
        output = net(temp_data)
        predictions = nd.argmax(output,axis=1)
        acc.update(preds = predictions, labels = temp_label)
    
    return acc.get()[1]

def getTestData(test_iter):

    result = []
    for data in test_iter:
        result.append(data)
    
    return result[0]

def getValiData(vali_iter):

    result = []
    for data in vali_iter:
        result.append(data)
    
    return result[0]

if __name__ == '__main__':

    ## List File Path:
    data_dir = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/aligned_final/'
    path_out = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/Processed/mxnet_list/'
    batch_size  = 32
    num_classes = 8
    channel = 3
    size = 224
    epoch = 5

    ctx = mx.cpu()

    train_iter = mx.image.ImageIter(batch_size = batch_size, data_shape=(channel, size, size), label_width=1,
                                   path_imglist=path_out+'train.lst',path_root=data_dir)
    vali_iter = mx.image.ImageIter(batch_size = 475, data_shape=(channel, size, size), label_width=1,
                                   path_imglist=path_out+'vali.lst',path_root=data_dir)
    test_iter = mx.image.ImageIter(batch_size = 480, data_shape=(channel, size, size), label_width=1,
                                   path_imglist=path_out+'test.lst',path_root=data_dir)

    test_data = getTestData(test_iter)
    vali_data = getValiData(vali_iter)

    resnet18 = vision.resnet18_v2(pretrained=False,ctx=ctx)

    ## Initialization
    resnet18.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(resnet18.collect_params(), 'sgd', {'learning_rate': 0.1})

    #train_iter.reset()
    #test_iter.reset()
    
    moving_loss = 0
    smoothing_constant = .01

    for e in range(epoch):

        train_loss, train_acc, n = 0.0, 0.0, 0.0
        train_iter.reset()
        test_iter.reset()

        start = time()

        for data_ in train_iter:
            losses = []
            data = data_.data[0]
            label = data_.label[0]

            with autograd.record():
                outputs = [resnet18(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]

            for l in losses:
                l.backward()
            
            train_acc += sum([(yhat.argmax(axis=1)==y).sum().asscalar()
                              for yhat, y in zip(outputs, label)])
            train_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)
            n += batch_size
        
        test_acc = evaluate_accuracy(test_data, resnet18, ctx)
        print("Epoch %d. Loss: %.3f, Train acc %.2f, Test acc %.2f, Time %.1f sec" % (
            epoch, train_loss/n, train_acc/n, test_acc, time() - start
        ))






   

