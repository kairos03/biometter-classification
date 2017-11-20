# Copyright kairos03 2017. All Right Reserved.

import os

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt

# data path
raw_path = './raw_data'
data_path = './data'
contact_path = raw_path + '/contact'
separate_path = raw_path + '/separate'


def read_image(path):
    """ Read images from path and return fnames, images"""
    print(os.listdir('.'))
    fnames = os.listdir(path)
    fnames.sort()

    imgs = []
    for name in fnames:
        img = ndimage.imread(path+name, True, 'L')
        imgs.append(img)
    return fnames, imgs


def concatenated_image(path):
    """contact front and right data"""
    # read images
    f_fnames, f_images = read_image(path + '/front/')
    r_fnames, r_images = read_image(path + '/right/')

    # reshape (330, 640, 512) to (330, 1, 640*512)
    f_images = np.reshape(f_images, (len(f_images), 1, 640 * 512))
    r_images = np.reshape(r_images, (len(r_images), 1, 640 * 512))

    # concat two images (330, 1, 640*512) to (330, 2, 640*512)
    images = np.concatenate((f_images, r_images), 1)

    return images


def integrated_data_set(to_csv=False, to_pickle=False):
    data_set = pd.DataFrame(columns=['image', 'is_contacted'])

    c_images = concatenated_image(contact_path)
    s_images = concatenated_image(separate_path)

    images = np.concatenate((c_images, s_images))

    for x in images:
        print(type(x))
        print(x)

    images = [x for x in images]
    print(type(images[0]))
    print(images[0])
    labels = np.concatenate((np.ones(c_images.shape[0], ), np.zeros(s_images.shape[0], )))

    data_set['image'] = images
    data_set['is_contacted'] = labels

    # TODO: bug fix; when saving dataframe to csv file, images ndarrays converted skipped string
    if to_csv:
        data_set.to_csv(data_path+'/input_data.csv', float_format='%.5f')

    if to_pickle:
        data_set.to_pickle(data_path+'/input_data.pkl', compression='gzip')

    return data_set


def make_train_and_test_set(test_ratio=0.2, to_pickle=False):
    data = read_whole_data()

    # shuffle
    data = data.sample(frac=1)

    # separate set
    pivot = int(data.shape[0] * test_ratio)

    train, test = data[pivot:], data[:pivot]

    if to_pickle:
        train.to_pickle(data_path+'/train_input_data.pkl', compression='gzip')
        test.to_pickle(data_path+'/test_input_data.pkl', compression='gzip')

    return train, test


def read_whole_data():
    return pd.read_pickle(data_path+'/input_data.pkl', compression='gzip')


def read_train_and_test_data():
    train = pd.read_pickle(data_path+'/train_input_data.pkl', compression='gzip')
    test = pd.read_pickle(data_path+'/test_input_data.pkl', compression='gzip')
    return train, test


def main():
    pass
    # save data set to pickle
    # data_set = integrated_data_set(to_pickle=True)

    # read whole data set
    # data_set = read_whole_data()

    # sample image plot
    # for i in [0, 600]:
    #     sample = data_set['image'][i]
    #     data = np.reshape(sample[0], (640, 512))
    #
    #     plt.imshow(data)
    #     plt.show()

    # make train, test set
    # train, test = make_train_and_test_set(to_pickle=True)

    # read train, test set
    # train, test = read_train_and_test_data()
    #
    # print(train.head())
    # print(test.head())


if __name__ == '__main__':
    main()
