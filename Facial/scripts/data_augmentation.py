
# This part is implemented using Keras

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import matplotlib.pylab as plt
import numpy as np
import scipy.io
from PIL import Image
import random

def demon():
    datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            cval=0,
            channel_shift_range=0,
            vertical_flip=False)

    test_img_path = "F:\AffectNet\\result\\no_face\\0be9c9c53fd9b7d58d300744821ea716ecafd71db78ce1da18558325.JPG"
    img = load_img(test_img_path)

    def random_crop(img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]


    x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
    x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='F:\AffectNet\\result\\test_aug', save_prefix='test', save_format='jpeg'):
        i += 1
        if i > 50:
            break  # 否则生成器会退出循环

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

def data_aug():
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        cval=0,
        channel_shift_range=0,
        vertical_flip=False)

    #x_test_path ='F:\\fer2013\\x_test.mat'
    #x_test = scipy.io.loadmat(x_test_path)['x_test']
    #x_test = x_test.reshape((len(x_test),48,48,1))

    test_img_path = "F:\AffectNet\\result\\no_face\\0be9c9c53fd9b7d58d300744821ea716ecafd71db78ce1da18558325.JPG"
    test_img = load_img(test_img_path)
    test_img = img_to_array(test_img)

    #print(test_img.shape)
    crop_ = random_crop(test_img,(1,1))
    print(crop_.shape)

    img = Image.fromarray(crop_, 'RGB')
    img.save('test.png')
    img.show()

    #test_img = x_test[1]
    #test_img = test_img.reshape((1,) + test_img.shape)
    #print(test_img.shape)


    # i = 0
    # for batch in datagen.flow(test_img, batch_size=1,
    #                           save_to_dir='F:\\fer2013\\test_aug\\test_1', save_prefix='test', save_format='jpg'):
    #     i += 1
    #     if i > 50:
    #         break  # 否则生成器会退出循环
    #


def random_crop_(image, crop_shape, padding=None):
    oshape = image.size

    if padding:
        oshape_pad = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
        img_pad = Image.new("RGB", (oshape_pad[0], oshape_pad[1]))
        img_pad.paste(image, (padding, padding))

        nh = random.randint(0, oshape_pad[0] - crop_shape[0])
        nw = random.randint(0, oshape_pad[1] - crop_shape[1])
        image_crop = img_pad.crop((nh, nw, nh + crop_shape[0], nw + crop_shape[1]))

        return image_crop
    else:
        print("WARNING!!! nothing to do!!!")
        return image

# def DataAugExample():
#     data_gen =

if __name__ == "__main__":
    #image_path = "F:\AffectNet\\result\\no_face\\0be9c9c53fd9b7d58d300744821ea716ecafd71db78ce1da18558325.JPG"

    image_path = "C:\\Users\Jiaming Nie\Documents\GitHub\DeepGlint-Work\Facial\datasets\\test\\0\\00617.jpg"
    image_src = Image.open(image_path)

    #print(image_src.shape)
    crop_width = image_src.size[0] - 3
    crop_height = image_src.size[1] - 3
    image_dst_crop = random_crop_(image_src, [crop_width, crop_height], padding=10)

    plt.figure()
    plt.subplot(221)
    plt.imshow(image_src)
    plt.title("oringin image")
    plt.subplot(222)
    plt.imshow(image_dst_crop)
    plt.title("crop image")
    plt.show()