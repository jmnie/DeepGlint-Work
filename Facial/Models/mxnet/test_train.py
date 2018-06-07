from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
mx.random.seed(1)
from mxnet.gluon.model_zoo import vision

def getNetwork(nb_class):
    resnet_18 = vision.resnet18_v2(pretrained=False,classes=nb_class)
    return resnet_18

batch_size = 16
CHANNEL = 3
SIZE = 224
EPOCH = 5
num_outputs = 8
std = 255
mean = 127

path_out = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/Processed/mxnet_list/'
data_dir = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/aligned_final/'
mean_array = nd.array([mean,mean,mean])
std_array = nd.array([std,std,std])

def normalization(image):
    image = mx.image.color_normalize(image,mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    #ormalized = mx.image.color_normalize(image,mean=mx.nd.array([0.485, 0.456, 0.406]),
    #                                 std=mx.nd.array([0.229, 0.224, 0.225]))
    return image

aug_list = mx.image.CreateAugmenter(data_shape=(3,224,224), resize=0, rand_crop=True, rand_resize=False, rand_mirror=True, mean=True, std=True, brightness=0.1, contrast=0, saturation=0, pca_noise=0, inter_method=2)

train_iter = mx.image.ImageIter(batch_size = batch_size, data_shape=(CHANNEL, SIZE, SIZE), 
                                label_width=1,
                                path_imglist=path_out+'test.lst',
                                path_root=data_dir,
                                #rand_mirror=True,
                                aug_list = aug_list,
                                shuffle = True)
                                   

test_iter = mx.image.ImageIter(batch_size = batch_size, data_shape=(CHANNEL, SIZE, SIZE), 
                               label_width=1,
                               path_imglist=path_out+'vali.lst',
                               path_root=data_dir,
                               shuffle = False)

def evaluate_accuracy(data_iterator, net, ctx):
    acc = mx.metric.Accuracy()
    for batch in enumerate(data_iterator):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

for i, batch in enumerate(train_iter):
    break
#print(i,batch.data[0])
#print(normalization(batch.data[0]).shape)

net = getNetwork(8)
ctx = mx.cpu()
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
#print(net)

epochs = 2
smoothing_constant = .01


for e in range(epochs):
    moving_loss = 0
    for i, batch in enumerate(train_iter):
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
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
            
    test_accuracy = evaluate_accuracy(test_iter, net, ctx)
    train_accuracy = evaluate_accuracy(train_iter, net, ctx)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))