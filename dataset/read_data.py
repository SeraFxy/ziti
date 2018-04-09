#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import random
import tensorflow as tf
from tensorflow.python.framework import ops  
from tensorflow.python.framework import dtypes  


dir_path, _ = os.path.split(os.path.realpath(__file__))

def read_labeled_image_list(images_dir):
    
    folders = [folder for _, folder, _ in os.walk(images_dir) if folder][0]
    
    filenames = []
    labels = []
    for index, folder in enumerate(folders):
        label = [0] * len(folders)
        image_dir = os.path.join(images_dir, folder)
        filename = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f[0] != '.']
        filenames += filename
        label[index] = 1.0
        labels += [label] * len(filename) 
        
    return filenames, labels, folders

def parse_function(filenames, label):

    file_contents = tf.read_file(filenames)
    example = tf.image.decode_png(file_contents, channels=3)
    example = tf.cast(tf.image.resize_images(example, [128, 128]), tf.uint8) 
    example = tf.image.per_image_standardization(example)
    return example, label

def read_data(batch_size):

    with tf.name_scope('input_pipeline'):
        filenames, labels, annotation = read_labeled_image_list(os.path.join(dir_path, 'images'))
        
        instances = list(zip(filenames, labels))
        random.shuffle(instances)
        filenames, labels = list(zip(*instances))
        filenames, labels = list(filenames), list(labels)
        
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(parse_function)

        dataset = dataset.shuffle(100).batch(batch_size).repeat()

        return dataset, annotation


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    
    batch_size = 8
    data_batch, annotation = read_data(batch_size)
    iterator = data_batch.make_initializable_iterator()
    # iterator = data_batch.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    
    # with tf.Session() as sess: 
    #     sess.run(iterator.initializer)
        
    #     for i in xrange(1):            
    #         image, label = sess.run([image_batch, label_batch])  
    #         for j in xrange(20):
    #             plt.imshow(image[j,:,:,:])
    #             print annotation[np.where(label[j]==1)[0][0]]
    #             plt.show()
    
        
