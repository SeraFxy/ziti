#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt 
import tensorflow as tf
from dataset.generator import read_text, draw_font
from nnets.vgg import vgg


def generator_images(texts, fonts):
    
    for text in texts:
        for font in fonts:
            image, _ = draw_font(text, font, mode='test')
            yield image, text

def show_errors(error_infos, fonts):

    
    length = len(error_infos)
    labels = len(fonts)
    print(fonts)
    for i in range(length):
        text, pred = error_infos[i]
        '''pred是错误的结果列表'''
        pred = [round(a, 5) for a in pred]
        index = pred.index(max(pred))
        for j in range(labels):
            axis = plt.subplot(length, labels, i * labels + j + 1)
            axis.axis('off')
            font = fonts[j]
            image, _ = draw_font(text, font, mode='test')
            if index == j:
                plt.title(str(pred))
            plt.imshow(image)
    plt.show()



def run():

    file_name = u'test.txt'
    # file_name = u'dataset/中国汉字大全.txt'
    texts = read_text(file_name)

    fonts_dir = os.path.join('dataset', 'fonts')
    fonts = [os.path.join(os.getcwd(), fonts_dir, path) for path in os.listdir(fonts_dir)] 

    images_gen = generator_images(texts, fonts)

    inputs = tf.placeholder(tf.float32, shape = [None, None, 3])
    example = tf.cast(tf.image.resize_images(inputs, [128, 128]), tf.uint8) 
    example = tf.image.per_image_standardization(example)
    example = tf.expand_dims(example, 0)
    outputs = vgg(example, 5, 1.0)

    sess = tf.Session()
    restorer = tf.train.Saver()
    restorer.restore(sess, 'models/vgg.ckpt')

    error = 0
    error_texts = []
    for index, info in enumerate(images_gen):
        
        image, text = info
        image = np.asarray(image)
        pred = sess.run(outputs, feed_dict={inputs:image})
        pred = np.squeeze(pred)
        label = np.squeeze(np.where(pred==np.max(pred)))
        if index%5  != label:
            error_texts.append((text, pred.tolist()))
            error += 1
            

    print('test num: {}, error num: {}, acc: {}'.format(index + 1, error, 1 - float(error) / index))
    show_errors(error_texts, fonts)
    

if __name__ == '__main__':
    run()

