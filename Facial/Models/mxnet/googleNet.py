# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""References:
Szegedy, Christian, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir
Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. "Going deeper
with convolutions." arXiv preprint arXiv:1409.4842 (2014).
"""

import mxnet as mx

def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    
    ## Add Batch Normalization
    bn = mx.symbol.BatchNorm(data = conv,axis = 1, eps = 0.0001,momentum = 0.9,name = 'bn_%s%s' %(name, suffix))
    
    act = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%s%s' %(name, suffix))
    return act

def InceptionFactory(data, num_1x1, num_3x3red, num_3x3, num_d5x5red, num_d5x5, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd5x5r = ConvFactory(data=data, num_filter=num_d5x5red, kernel=(1, 1), name=('%s_5x5' % name), suffix='_reduce')
    cd5x5 = ConvFactory(data=cd5x5r, num_filter=num_d5x5, kernel=(5, 5), pad=(2, 2), name=('%s_5x5' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd5x5, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def get_symbol(num_classes = 1000, **kwargs):
    data = mx.sym.Variable("data")
    conv1 = ConvFactory(data, 64, kernel=(7, 7), stride=(2,2), pad=(3, 3), name="conv1")
    pool1 = mx.sym.Pooling(conv1, kernel=(3, 3), stride=(2, 2), pool_type="max")
    conv2 = ConvFactory(pool1, 64, kernel=(1, 1), stride=(1,1), name="conv2")
    conv3 = ConvFactory(conv2, 192, kernel=(3, 3), stride=(1, 1), pad=(1,1), name="conv3")
    pool3 = mx.sym.Pooling(conv3, kernel=(3, 3), stride=(2, 2), pool_type="max")

    in3a = InceptionFactory(pool3, 64, 96, 128, 16, 32, "max", 32, name="in3a")
    in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, "max", 64, name="in3b")
    pool4 = mx.sym.Pooling(in3b, kernel=(3, 3), stride=(2, 2), pool_type="max")
    in4a = InceptionFactory(pool4, 192, 96, 208, 16, 48, "max", 64, name="in4a")
    in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, "max", 64, name="in4b")
    in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, "max", 64, name="in4c")
    in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, "max", 64, name="in4d")
    in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, "max", 128, name="in4e")
    pool5 = mx.sym.Pooling(in4e, kernel=(3, 3), stride=(2, 2), pool_type="max")
    in5a = InceptionFactory(pool5, 256, 160, 320, 32, 128, "max", 128, name="in5a")
    in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, "max", 128, name="in5b")
    pool6 = mx.sym.Pooling(in5b, kernel=(2, 2), stride=(1,1), global_pool=True, pool_type="avg")
    flatten = mx.sym.Flatten(data=pool6)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax

def Training():
    nb_class = 7
    EPOCH = 5
    CHANNEL = 1
    SIZE = 48
    batch_size = 16
    lr = 0.1
    symbol = get_symbol(nb_class)
    model = mx.mod.Module(symbol=symbol,
                          context = mx.cpu(),
                          data_names=['data'],
                          label_names=['softmax_label'])
    
    path_out = '/media/jiaming/Seagate Backup Plus Drive/fer2013/mxnet_file/'

#     aug_list = mx.image.CreateAugmenter(data_shape=(CHANNEL,SIZE,SIZE),
#                            rand_crop=False, 
#                            rand_resize=False, 
#                            rand_mirror=False, 
#                            mean=True, 
#                            std=True, 
#                            brightness=0.1, 
#                            contrast=0.2, 
#                            saturation=0.1)
    #aug_seq = [mx.image.ColorJitterAug(0.1,0.2,0.1),mx.image.HueJitterAug(0.1)]
    ## Fixed Crop:mx.image.fixed_crop(src, x0, y0, w, h, size=None, interp=2)
    
    train_iter = mx.io.ImageRecordIter(                
                    path_imgrec = path_out + 'train.rec', # The target record file.
                    path_imgidx = path_out + 'train.idx',
                    preprocess_threads = 10,
                    #aug_seq = aug_seq,
                    data_shape=(CHANNEL, SIZE, SIZE), # Output data shape; 227x227 region will be cropped from the original image.
                    batch_size=batch_size, # Number of items per batch.
                    #resize=224, # Resize the shorter edge to 256 before cropping.
                    rand_crop = True,
                    shuffle = True,
                    mean_r = 128,
                    mean_g = 128,
                    mean_b = 128,
                    std_r = 256,
                    std_g = 256,
                    std_b = 256,
        
#                     mean_r = 124.16,
#                     mean_g = 116.736,
#                     mean_b = 103.936,
#                     std_r = 58.624,
#                     std_g = 57.344,
#                     std_b = 57.6,
                    #scale = 0.00392, #1/255
                    random_mirror = True,
        )

    vali_iter = mx.io.ImageRecordIter(
                    path_imgrec = path_out + 'test.rec', # The target record file.
                    path_imgidx = path_out + 'test.idx',
                    preprocess_threads = 10,
                    data_shape=(CHANNEL,SIZE,SIZE), # Output data shape; 227x227 region will be cropped from the original image.
                    batch_size = batch_size, # Number of items per batch.
                    rand_crop = False,
                    shuffle = False,
                    Amean_r = 128,
                    mean_g = 128,
                    mean_b = 128,
                    std_r = 256,
                    std_g = 256,
                    std_b = 256,
#                     mean_r = 124.16,
#                     mean_g = 116.736,
#                     mean_b = 103.936,
#                     std_r = 58.624,
#                     std_g = 57.344,
#                     std_b = 57.6,
            )
    
    ### See the data:
    
    for batch in train_iter:
        break
    print(batch.data[0].shape)
    
    
    
    # allocate memory given the input data and label shapes
    model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    # initialize parameters by uniform random numbers
    model.init_params(initializer=mx.init.Xavier(magnitude=1))
    # use SGD with learning rate 0.1 to train
    model.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': 0.1, 'wd': 0.0001})
    # use accuracy as the metric
    metric = mx.metric.create('acc')
    metric_1 = mx.metric.create('acc')
    # train 5 epochs, i.e. going over the data iter one pass
    
    #### Cross Entropy loss
    ce = mx.metric.CrossEntropy()
    ce_1 = mx.metric.CrossEntropy()
    
    for epoch in range(EPOCH):
        train_iter.reset()
        metric.reset()
        ce.reset()
        for batch in train_iter:
            model.forward(batch, is_train=True)       # compute predictions
            model.update_metric(metric, batch.label)  # accumulate prediction accuracy
            model.update_metric(ce, batch.label)
            model.backward()                          # compute gradients
            model.update()                            # update parameters
        
        vali_iter.reset()
        metric_1.reset()
        ce_1.reset()
        for batch in vali_iter:
            model.forward(batch, is_train=False)
            model.update_metric(metric_1, batch.label)
            model.update_metric(ce_1, batch.label)
        
        print('Epoch %d, Training_acc %s, Vali_acc %s, Training_Loss %s, Vali_Loss %s' % (epoch, metric.get()[1], metric_1.get()[1], ce.get()[1],ce_1.get()[1]))
        #print('Epoch %d, Vali_acc %s, Vali_Loss %s,' % (epoch, metric_1.get()[1], ce_1.get()[1]))

#         model.fit(
#         train_data = train_iter,
#         eval_data = vali_iter,
#         optimizer = 'sgd',
#         optimizer_params = {'learning_rate':0.01, 'momentum': 0.9},
#         initializer = mx.init.Xavier(magnitude=1),
#         eval_metric = 'accuracy',
#         num_epoch = EPOCH)
    
if __name__ == '__main__':
    #Training()
    import math
    print(-math.log(0.7486))