
import tensorflow as tf
from nnets.vgg import vgg

import numpy as np
import  tkinter
from tkinter.filedialog import askopenfilename
from tkinter import *
import tkinter.font as tkFont
from PIL import Image, ImageFont, ImageDraw,ImageTk
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 不使用GPU


def selectPath():
    path_ = askopenfilename()
    path.set(path_)
    '''将选择的路径结果赋值到path中'''
def end():
    root.destroy()
    '''结束进程'''

inputs = tf.placeholder(tf.float32, shape=[None, None, 3])
example = tf.cast(tf.image.resize_images(inputs, [128, 128]), tf.uint8)
example = tf.image.per_image_standardization(example)
example = tf.expand_dims(example, 0)
output = vgg(example, 5, 1.0)
sess = tf.Session()
tf.train.Saver().restore(sess, 'models/vgg.ckpt')
'''导入模型'''


root = Tk()
root.geometry('570x185+500+400')
'''设置框体的大小'''
path = StringVar()
image_frame = Frame(root)
image_file = im = image_label = None
def create_image_label():
    '''创建图片lable和字体识别lable'''
    data = Image.open(path.get())
    '''跟文件名没有关系'''
    pred = sess.run(output, feed_dict={inputs: data})
    pred = np.squeeze(pred)
    pred = pred.tolist()
    fenlei=''
    index = pred.index(max(pred))
    '''获取最大值进行判断'''
    if index == 0:
        fenlei = '欧体'
    elif index == 1:
        fenlei = '赵体'
    elif index == 2:
        fenlei = '柳体'
    elif index == 3:
        fenlei = '魏碑'
    elif index == 4:
        fenlei = '颜体'

    global image_file, im, image_label
    image_file = Image.open(path.get())
    out =image_file .resize((100, 100), Image.ANTIALIAS)
    '''设置图片大小'''
    im = ImageTk.PhotoImage(out)
    image_label = Label(image_frame,image = im)
    image_label.grid(row = 2, column = 0, sticky = NW, pady = 8, padx = 20)
    image_label2 = Label(image_frame,text = fenlei,font=("黑体", 30, "bold"),bg='white')
    image_label2.grid(row=2, column=1,  pady=40, padx=20)
    Label(image_frame, text="目标图片：", width=15).grid(row=1, column=0)
    Label(image_frame, text="字体识别为：", width=15).grid(row=1, column=1)
    '''框体位置大小等信息'''

Label(image_frame,text = "目标路径:",width = 18).grid(row = 0, column = 0)
Entry(image_frame, width=49,bg='white',textvariable = path).grid(row = 0, column = 1)
Button(image_frame, text = "路径选择", command = selectPath).grid(row = 0, column = 2)
button = Button(image_frame,text='确认路径',anchor = 'center',command = create_image_label)
button.grid(row = 1, column = 2, sticky = NW, pady = 8, padx = 20)
Button(image_frame, text = "结束进程", command = end).grid(row = 2, column = 2)
image_frame.pack()
root.mainloop()
'''主界面参数及其设置'''













