from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
mx.random.seed(1)
from mxnet.gluon.model_zoo import vision

def getNetwork(nb_class):
    resnet_ = vision.resnet34_v2(pretrained=False,classes=nb_class)
    #resnet_18.collect_params().initialize(mx.init.Xavier(magnitude=0.5), ctx=ctx)
    return resnet_

ctx = mx.cpu()

batch_size = 16
CHANNEL = 3
SIZE = 224
EPOCH = 5
num_outputs = 8
std = 255
mean = 127.5

path_out = '/home/jiaming/code/github/DeepGlint-Work/Facial/scripts/rec_file/'
# data_dir = '/train/trainset/1/Manually_Annotated_Images/'
# mean_array = nd.array([mean,mean,mean])
# std_array = nd.array([std,std,std])

aug_list = mx.image.CreateAugmenter(data_shape=(CHANNEL,SIZE,SIZE),
                           rand_crop=True, 
                           rand_resize=False, 
                           rand_mirror=True, 
                           mean=True, 
                           std=True, 
                           brightness=0.1, 
                           contrast=0.1, 
                           saturation=0)

train_iter = mx.io.ImageRecordIter(
                path_imgrec = path_out + 'test.rec', # The target record file.
                path_imgidx = path_out + 'test.idx',
                preprocess_threads = 5,
                #aug_seq = aug_list,
                data_shape=(3, 224, 224), # Output data shape; 227x227 region will be cropped from the original image.
                batch_size=batch_size, # Number of items per batch.
                resize=256, # Resize the shorter edge to 256 before cropping.
                rand_crop = True,
                shuffle = True,
                scale = 0.00392,
                random_mirror = True,
    )

vali_iter = mx.io.ImageRecordIter(
                path_imgrec = path_out + 'vali.rec', # The target record file.
                path_imgidx = path_out + 'vali.idx',
                preprocess_threads = 5,
                data_shape=(3, 224, 224), # Output data shape; 227x227 region will be cropped from the original image.
                batch_size = batch_size, # Number of items per batch.
                rand_crop = True,
                shuffle = False,
                scale = 0.00392)

for i, batch in enumerate(train_iter):
    break
    #print(batch.data[0])
print(batch.data[0])

net = getNetwork(8)
net.collect_params().initialize(mx.init.Xavier(magnitude=1), ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .1})

def evaluate_accuracy(data_iterator, net):
    data_iterator.reset()
    acc = mx.metric.Accuracy()
    for i, batch in enumerate(data_iterator):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

def vali_loss_cal(data_iter,net):
    data_iter.reset()
    #moving_loss = 0
    smoothing_constant = .01
    for i, batch in enumerate(train_iter):
        #print(data.shape)
        #print(label.shape)
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                    else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
    
    return moving_loss

epochs = 200
smoothing_constant = .01

train_acc_result = []
vali_acc_result = []
train_loss_res = []
vali_loss_res = []

for e in range(epochs):
    #train_iter.reset()
    train_iter.reset()
    for i, batch in enumerate(train_iter):
        #print(data.shape)
        #print(label.shape)
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
                       
    train_accuracy = evaluate_accuracy(train_iter, net)
    vali_accuracy = evaluate_accuracy(vali_iter, net)
    vali_loss = vali_loss_cal(vali_iter,net)

    train_acc_result.append(train_accuracy)
    vali_acc_result.append(vali_accuracy)
    train_loss_res.append(moving_loss)
    vali_loss_res.append(vali_loss)

    print("Epoch %s. Loss: %s, Train_acc %s, Vali_acc %s, Vali_loss %s" % (e, moving_loss, train_accuracy, vali_accuracy, vali_loss))   

train_acc_result = np.array(train_acc_result)
vali_acc_result = np.array(vali_acc_result)
train_loss_res = np.array(train_loss_res)
vali_loss_res = np.array(vali_loss_res)

path = '/home/jiaming/code/github/DeepGlint-Work/Facial/Models/mxnet/result/'
np.savetxt(path+'train_acc.txt',train_acc_result)
np.savetxt(path+'vali_acc.txt',vali_acc_result)
np.savetxt(path+'train_loss.txt',train_loss_res)
np.savetxt(path+'vali_loss.txt',vali_loss_res)