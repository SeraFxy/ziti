
import tensorflow as tf
from nnets.vgg import vgg

import numpy as np
import  tkinter
from tkinter.filedialog import askopenfilename
from tkinter import *
import tkinter.font as tkFont
from PIL import Image, ImageFont, ImageDraw,ImageTk
import  os


def run(path):


    data = Image.open(path)
    '''跟文件名没有关系'''
    pred = sess.run(output, feed_dict={inputs: data})
    pred = np.squeeze(pred)
    pred = pred.tolist()
    fenlei=''
    index = pred.index(max(pred))
    '''获取最大值进行判断'''
    if index == 0:
        fenlei = '草体'
    elif index == 1:
        fenlei = '楷体'
    elif index == 2:
        fenlei = '篆体'
    elif index == 3:
        fenlei = '隶书'
    return fenlei

run('C:\\Users\Beyond\Desktop\\a.jpg')











