import tensorflow as tf
from nnets.vgg import vgg

import numpy as np
import  tkinter
from tkinter.filedialog import askopenfilename
from tkinter import *
from PIL import Image, ImageFont, ImageDraw

inputs = tf.placeholder(tf.float32, shape = [None, None, 3])
example = tf.cast(tf.image.resize_images(inputs, [128, 128]), tf.uint8)
example = tf.image.per_image_standardization(example)
example = tf.expand_dims(example, 0)
output = vgg(example, 5, 1.0)
sess=tf.Session()
tf.train.Saver().restore(sess, 'models/vgg.ckpt')
print("Model restored.")

def selectPath():
    path_ = askopenfilename()
    path.set(path_)
def end():
    root.destroy()
root = Tk()
path = StringVar()
Label(root,text = "目标路径:").grid(row = 0, column = 0)
Entry(root, width=100,bg='red',textvariable = path).grid(row = 0, column = 1)
Button(root, text = "路径选择", command = selectPath).grid(row = 0, column = 2)
Button(root, text = "确定选择", command = end).grid(row = 1, column = 2)
root.mainloop()

data = Image.open(path.get())
'''跟文件名没有关系'''
pred = sess.run(output, feed_dict={inputs: data})
pred = np.squeeze(pred)
pred=pred.tolist()
index =pred.index(max(pred))
print (index)
