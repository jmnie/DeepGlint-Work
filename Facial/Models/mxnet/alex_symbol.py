import mxnet as mx
import numpy as np

def get_symbol(num_classes, dtype='float32', **kwargs):
    input_data = mx.sym.Variable(name="data")
    if dtype == 'float16':
        input_data = mx.sym.Cast(data=input_data, dtype=np.float16)
    # stage 1
    conv1 = mx.sym.Convolution(name='conv1',
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.sym.Activation(data=conv1, act_type="relu")
    lrn1 = mx.sym.LRN(data=relu1, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool1 = mx.sym.Pooling(
        data=lrn1, pool_type="max", kernel=(3, 3), stride=(2,2))
    # stage 2
    conv2 = mx.sym.Convolution(name='conv2',
        data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.sym.Activation(data=conv2, act_type="relu")
    lrn2 = mx.sym.LRN(data=relu2, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool2 = mx.sym.Pooling(data=lrn2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 3
    conv3 = mx.sym.Convolution(name='conv3',
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.sym.Activation(data=conv3, act_type="relu")
    conv4 = mx.sym.Convolution(name='conv4',
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.sym.Activation(data=conv4, act_type="relu")
    conv5 = mx.sym.Convolution(name='conv5',
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.sym.Activation(data=conv5, act_type="relu")
    pool3 = mx.sym.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.sym.Flatten(data=pool3)
    fc1 = mx.sym.FullyConnected(name='fc1', data=flatten, num_hidden=4096)
    relu6 = mx.sym.Activation(data=fc1, act_type="relu")
    dropout1 = mx.sym.Dropout(data=relu6, p=0.5)
    # stage 5
    fc2 = mx.sym.FullyConnected(name='fc2', data=dropout1, num_hidden=4096)
    relu7 = mx.sym.Activation(data=fc2, act_type="relu")
    dropout2 = mx.sym.Dropout(data=relu7, p=0.5)
    # stage 6
    fc3 = mx.sym.FullyConnected(name='fc3', data=dropout2, num_hidden=num_classes)
    if dtype == 'float16':
        fc3 = mx.sym.Cast(data=fc3, dtype=np.float32)
    softmax = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
    return softmax

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def plotcurve(acc_train,acc_vali,loss_train,loss_vali):
    plt.style.use('ggplot')
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.set_title('Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epochs')
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.set_title('Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    #ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    #ax3.set_title('Time')
    #ax3.set_ylabel('Seconds')
    
    ax1.plot(acc_vali, label='Vali')
    ax1.plot(acc_train, label='Train')
    ax2.plot(loss_vali, label='Vali')
    ax2.plot(loss_train, label='Train')

    ax1.legend()
    ax2.legend()
    #ax3.bar(np.arange(len(results)), [x[1] for x in results],
    #        align='center')
    plt.tight_layout()
    plt.show()
    
def cm_plot(path_out,net):
    
    LABELS = ["Neutral",
              "Happy",
              "Sad",
              "Surprise",
              "Fear",
              "Disgust",
              "Anger",
              "Contempt"]
    num_output = len(LABELS)
    
    test_iter = mx.io.ImageRecordIter(                
            path_imgrec = path_out + 'new_vali.rec', # The target record file.
            path_imgidx = path_out + 'new_vali.idx',
            preprocess_threads = 4,
            #aug_seq = aug_seq,
            data_shape = (3, 224, 224), # Output data shape; 227x227 region will be cropped from the original image.
            batch_size = 16, # Number of items per batch.
            resize = 224, # Resize the shorter edge to 256 before cropping.
            rand_crop = False,
            shuffle = False,
            mean_r = 128,
            mean_g = 128,
            mean_b = 128,
            std_r = 256,
            std_g = 256,
            std_b = 256,
            random_mirror = False,
        )
    
    
    
    predictions = []
    label = []
    for batch in test_iter:
        net.forward(batch)
        prob = net.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        
        for j in range(len(batch.label[0])):
            temp_label = batch.label[0].asnumpy()
            label.append(temp_label[j])
            
        for i in range(len(prob)):
            predictions.append(np.argsort(prob[i])[::-1][0])
            
        
    predictions = np.array(predictions)
    y_test = np.array(label)

    print(y_test.shape,predictions.shape)
    print("")
    print("Accuracy: {}%".format(100*metrics.accuracy_score(y_test, predictions)))
    print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
    print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))
    
    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    # confusion_matrix = metrics.confusion_matrix(one_hot(y_test), predictions_one_hot)
    print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

    print("")
    print("Confusion matrix (normalised to % of total test data):")
    print(normalised_confusion_matrix)
    print("Note: training and testing data is not equally distributed amongst classes, ")
    print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

    # Plot Results: 
    width = 8
    height = 8
    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix, 
        interpolation='nearest', 
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(num_output)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def Training():
    nb_class = 8
    EPOCH = 1
    CHANNEL = 3
    SIZE = 224
    batch_size = 16
    lr = 0.1
    
    symbol = get_symbol(nb_class)
    model = mx.mod.Module(symbol=symbol,
                          context = mx.gpu(0),
                          data_names=['data'],
                          label_names=['softmax_label'])#,context=mx.gpu())
    
    #path_out = '/train/execute/AffectNet/dict_files/'
    path_out = '/train/execute/AffectNet/dict_files/'
#     aug_list = mx.image.CreateAugmenter(data_shape=(CHANNEL,SIZE,SIZE),
#                            rand_crop=False, 
#                            rand_resize=False, 
#                            rand_mirror=False, 
#                            mean=True, 
#                            std=True, 
#                            brightness=0.1, 
#                            contrast=0.2, 
#                            saturation=0.1)
    aug_seq = [mx.image.ColorJitterAug(0.1,0.2,0.1),mx.image.HueJitterAug(0.1)]
    ## Fixed Crop:mx.image.fixed_crop(src, x0, y0, w, h, size=None, interp=2)
    
    train_iter = mx.io.ImageRecordIter(                
                    path_imgrec = path_out + 'new_train_7.rec', # The target record file.
                    path_imgidx = path_out + 'new_train_7.idx',
                    preprocess_threads = 15,
                    #aug_seq = aug_seq,
                    data_shape=(3, 224, 224), # Output data shape; 227x227 region will be cropped from the original image.
                    batch_size=batch_size, # Number of items per batch.
                    resize=224, # Resize the shorter edge to 256 before cropping.
                    rand_crop = True,
                    shuffle = True,
                    mean_r = 128,
                    mean_g = 128,
                    mean_b = 128,
                    std_r = 256,
                    std_g = 256,
                    std_b = 256,
                    random_mirror = True,
        )

    vali_iter = mx.io.ImageRecordIter(
                    path_imgrec = path_out + 'new_vali.rec', # The target record file.
                    path_imgidx = path_out + 'new_vali.idx',
                    preprocess_threads = 15,
                    data_shape=(3, 224, 224), # Output data shape; 227x227 region will be cropped from the original image.
                    batch_size = batch_size, # Number of items per batch.
                    rand_crop = False,
                    shuffle = False,
                    Amean_r = 128,
                    mean_g = 128,
                    mean_b = 128,
                    std_r = 256,
                    std_g = 256,
                    std_b = 256,
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
    model.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': lr, 'wd': 0.0001})
    # use accuracy as the metric
    metric = mx.metric.create('acc')
    metric_1 = mx.metric.create('acc')
    # train 5 epochs, i.e. going over the data iter one pass
    
    #### Cross Entropy loss
    ce = mx.metric.CrossEntropy()
    ce_1 = mx.metric.CrossEntropy()
    
    ### Loss and Accuracy
    acc_train = []
    acc_vali = [] 
    loss_train = [] 
    loss_vali = [] 
    
    
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
        acc_train.append(metric.get()[1])
        acc_vali.append(metric_1.get()[1])
        loss_train.append(ce.get()[1])
        loss_vali.append(ce_1.get()[1])
        #print('Epoch %d, Vali_acc %s, Vali_Loss %s,' % (epoch, metric_1.get()[1], ce_1.get()[1]))
    
    plotcurve(acc_train,acc_vali,loss_train,loss_vali)
    cm_plot(path_out,model)

if __name__ == '__main__':
    Training()