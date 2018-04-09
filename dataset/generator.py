#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys


import queue
import threading
from PIL import Image, ImageFont, ImageDraw
import numpy as np 
from tqdm import tqdm  

def draw_font(text, font, save_path=None, mode='train'):
    
    image_name = '{}{}.png'.format(save_path, text)
    
    if mode == 'train' and os.path.isfile(image_name):
        return

    im = Image.new("RGB", (256, 256), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(font, 128)
     
    dr.text((64, 64), text, font=font, fill="#000000")
    im_slice = np.asarray(im)[:,:,0]
    y, x = np.where(im_slice == 255)
    x_max, x_min, y_max, y_min = np.max(x), np.min(x), np.max(y), np.min(y)

    frame = 20
    box = (x_min + frame, y_min + frame, x_max - frame, y_max - frame)
    im = im.crop(box)
    return im, image_name
    


def generator(fonts, texts, consumer_num):

    with tqdm(total=len(fonts)*len(texts)) as counter:  
        for font in fonts:
            save_path = 'images/{}/'.format(font.split('.')[0])
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            for text in texts:
                font = os.path.join(os.getcwd(), 'fonts', font)
                result = draw_font(text, font, save_path)
                if result:
                    message.put(result)
                counter.update(1)
    for _ in range(consumer_num):
        message.put(None)

def writer():

    while True:
        msg = message.get()
        if msg:
            im, image_name = msg  
            im.save(image_name)
        else:
            break

def read_text(file_name):

    with open(file_name, encoding='UTF-8',errors='ignore') as f:
        texts = f.read().split(' ')
    return texts
    
def run():
    
    file_name = u'中国汉字大全.txt'
    texts = read_text(file_name)
    fonts = os.listdir('fonts')
    
    consumer_1 = threading.Thread(target=writer)
    consumer_2 = threading.Thread(target=writer)
    consumer_num = 2
    producer = threading.Thread(target=generator, args=(fonts, texts, consumer_num,))


    producer.start()
    consumer_1.start()
    consumer_2.start()
    message.join()

if __name__ == '__main__':

    message = queue.Queue(1000)
    run()    