#!/usr/bin/env python
# -*- coding: utf-8 -*-
import  tensorflow as tf
import  os
from nnets.vgg import  vgg


def predict(class_num,path,data):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 不使用GPU
        inputs = tf.placeholder(tf.float32, shape=[None, None, 3])
        example = tf.cast(tf.image.resize_images(inputs, [128, 128]), tf.uint8)
        example = tf.image.per_image_standardization(example)
        example = tf.expand_dims(example, 0)
        output = vgg(example, class_num, 1.0)
        sess = tf.Session()
        tf.train.Saver().restore(sess, path)
        pred = sess.run(output, feed_dict={inputs: data})
        sess.close()
        return pred




