import json
import numpy as np
import os 
from sklearn.model_selection import train_test_split
import mxnet as mx

def mapping_file(dataset_path,map_path):
    
    #crop_image = '/train/execute/AffectNet/Data/aligned/'
    dirs = os.listdir(dataset_path)
    
    with open('/home/jiaming/code/github/DeepGlint-Work/affect_dict.json','r') as fp:
        dataset_dict = json.load(fp)
    
    new_dataset_dict = dict()
    
    useless = [8,9,10]
    for subdir in dirs:
        dir_path = os.path.join(dataset_path,subdir)
        for file in os.listdir(dir_path):
            key = subdir + '/' + file
            
            if key in dataset_dict:
                if dataset_dict[key][0] not in useless:
                    new_dataset_dict[key] = dataset_dict[key]
              
    # with open('/train/execute/AffectNet/crop_AffectNet.json','w') as fp:
    #     json.dump(new_dataset_dict,fp)
    
    x_name = []
    y_label = []
    for key in new_dataset_dict:
        x_name.append(key)
        y_label.append(new_dataset_dict[key][0])
    
    y_label = np.array(y_label)
    x_name = np.array(x_name)
    
    x_remain, x_test, y_remain, y_test = train_test_split(x_name, y_label, test_size=0.01, random_state=50)
    x_train, x_vali, y_train, y_vali = train_test_split(x_remain, y_remain, test_size = 0.01, random_state=50)
    print(len(y_label))
    print(y_train.shape)
    print(y_vali.shape)
    print(y_test.shape)
    x_train = x_train.tolist()
    x_test = x_test.tolist()
    x_vali = x_vali.tolist()
    
    np.savetxt(map_path + '/y_train.txt',y_train)
    np.savetxt(map_path + '/y_test.txt',y_test)
    np.savetxt(map_path + '/y_vali.txt',y_vali)
    
    def writetxt(path,list_to_write):
        if os.path.exists(path):  
            os.remove(path)  
        file_write_obj = open(path, 'w')
        
        for i in range(len(list_to_write)):  
            file_write_obj.writelines(list_to_write[i])  
            file_write_obj.write('\n')  
        file_write_obj.close()  
    
    x_train_dict = dict()
    x_test_dict = dict()
    x_vali_dict = dict()
    
    for i in range(len(x_train)):
        x_train_dict[i] = x_train[i]
    
    for i in range(len(x_test)):
        x_test_dict[i] = x_test[i]
    
    for i in range(len(x_vali)):
        x_vali_dict[i] = x_vali[i]
        
    with open(map_path + '/x_train.json','w') as fp:
        json.dump(x_train,fp)
    with open(map_path + '/x_vali.json', 'w') as fp:
        json.dump(x_vali,fp)
    with open(map_path + '/x_test.json', 'w') as fp:
        json.dump(x_test,fp)
    #np.savetxt('/train/execute/AffectNet/Data/224_crop_mapping/x_train.txt',x_train)
    #np.savetxt('/train/execute/AffectNet/Data/224_crop_mapping/x_test.txt',x_test)
    #np.savetxt('/train/execute/AffectNet/Data/224_crop_mapping/x_vali.txt',x_vali)
    
    print("Done")

def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            #print("i:",i,"item:",item)
            line = '%d\t' % item[0]
            for j in item[2:]:
               line += '%f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)

def mxnet_makelst(mapping_dir,data_dir,out_path):
    x_train_path = mapping_dir + '/x_train.json'
    x_test_path = mapping_dir + '/x_test.json'
    x_vali_path = mapping_dir + '/x_vali.json'

    y_train_path = mapping_dir + '/y_train.txt'
    y_test_path = mapping_dir + '/y_test.txt'
    y_vali_path = mapping_dir + '/y_vali.txt'

    y_train = np.loadtxt(y_train_path)
    y_test = np.loadtxt(y_test_path)
    y_vali = np.loadtxt(y_vali_path)

    with open(x_train_path,'r') as fp:
        x_train = json.load(fp)
    
    with open(x_test_path,'r') as fp:
        x_test = json.load(fp)
    
    with open(x_vali_path,'r') as fp:
        x_vali = json.load(fp)
    
    train_list = []
    for i in range(len(x_train)):
        line = []
        line.append(i)
        #path = data_dir + x_train[i]
        path = x_train[i]
        line.append(path)
        label = y_train[i]
        line.append(label)
        train_list.append(line)
    
    test_list = []
    for i in range(len(x_test)):
        line = []
        line.append(i)
        #path = data_dir + x_test[i]
        path = x_test[i]
        line.append(path)
        label = y_test[i]
        line.append(label)
        test_list.append(line)
    
    vali_list = []
    for i in range(len(x_vali)):
        line = []
        line.append(i)
        #path = data_dir + x_vali[i]
        path = x_vali[i]
        line.append(path)
        label = y_vali[i]
        line.append(label)
        vali_list.append(line)
    
    ### Write List file
    path_out_train = out_path + 'train.lst'
    path_out_test = out_path + 'test.lst'
    path_out_vali = out_path + 'vali.lst'

    write_list(path_out_train,train_list)
    write_list(path_out_test,test_list)
    write_list(path_out_vali,vali_list)

if __name__ == "__main__":
    # mapping_dir = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/Processed/224_mapping/basic_emotion/'
    # data_dir = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/aligned_final/'
    # path_out = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/Processed/mxnet_list/'
    # mapping_file(data_dir,mapping_dir)
    # mxnet_makelst(mapping_dir,data_dir,path_out)

    data_iter = mx.io.ImageRecordIter(
                path_imgrec = "/home/jiaming/code/github/DeepGlint-Work/Facial/scripts/test.rec", # The target record file.
                path_imgidx = "/home/jiaming/code/github/DeepGlint-Work/Facial/scripts/test.idx",
                data_shape=(3, 224, 224), # Output data shape; 227x227 region will be cropped from the original image.
                batch_size=4, # Number of items per batch.
                resize=256, # Resize the shorter edge to 256 before cropping.
                rand_crop = True,
    )
    # You can now use the data_iter to access batches of images.
    batch = data_iter.next() # first batch.
    images = batch.data[0] # This will contain 4 (=batch_size) images each of 3x227x227.
    # process the images
    label = batch.label[0]

    data_iter.reset() # To restart the iterator from the beginning.

    print(images.shape)
    print(label.shape)
