#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf 

def conv_op(input_op, name, n_out, kh=3, kw=3, dh=1, dw=1):

    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=[kh, kw, n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(.0, shape=[n_out], dtype=tf.float32)
        bias = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, bias)
        activation = tf.nn.relu(z, name=scope)
        tf.summary.histogram('histogram', activation)
        return activation

def fc_op(input_op, name, n_out):

    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w', shape=[n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        tf.summary.histogram('histogram', activation)
        return activation

def mpool_op(input_op, name, kh=2, kw=2, dh=2, dw=2):

    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)

def vgg(input_op, class_num, keep_prob):

    with tf.name_scope('vgg'):

        conv1_1 = conv_op(input_op, name='conv1_1', n_out=64)
        conv1_2 = conv_op(conv1_1, name='conv1_2',  n_out=64)
        pool1 = mpool_op(conv1_2, name='pool1')

        conv2_1 = conv_op(pool1, name='conv2_1', n_out=128)
        conv2_2 = conv_op(conv2_1, name='conv2_2', n_out=128)
        pool2 = mpool_op(conv2_2, name='pool2')

        conv3_1 = conv_op(pool2, name='conv3_1', n_out=256)
        conv3_2 = conv_op(conv3_1, name='conv3_2', n_out=256)
        conv3_3 = conv_op(conv3_2, name='conv3_3', n_out=256)
        pool3 = mpool_op(conv3_3, name='pool3')

        conv4_1 = conv_op(pool3, name='conv4_1', n_out=512)
        conv4_2 = conv_op(conv4_1, name='conv4_2', n_out=512)
        conv4_3 = conv_op(conv4_2, name='conv4_3', n_out=512)
        pool4 = mpool_op(conv4_3, name='pool4')

        conv5_1 = conv_op(pool4, name='conv5_1', n_out=512)
        conv5_2 = conv_op(conv5_1, name='conv5_2', n_out=512)
        conv5_3 = conv_op(conv5_2, name='conv5_3', n_out=512)
        pool5 = mpool_op(conv5_3, name='pool5')

        shp = pool5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')

        fc6 = fc_op(resh1, name='fc6', n_out=2048)
        fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')

        fc7 = fc_op(fc6_drop, name='fc7', n_out=2048)
        fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc6_drop')

        fc8 = fc_op(fc7_drop, name='fc8', n_out=class_num)
        softmax = tf.nn.softmax(fc8)

        return softmax

