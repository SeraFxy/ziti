# coding:utf-8

from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from nnets.vgg import vgg
import numpy as np
from PIL import Image, ImageFont, ImageDraw,ImageTk
from Predict import predict


app = Flask(__name__)


'''导入模型'''
@app.route('/')
def about():
    return redirect(url_for('upload'))

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath,'static\\uploads',f.filename)  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        data = Image.open(upload_path)
        '''跟文件名没有关系'''
        pred = predict(4,'models/vgg.ckpt',data)
        pred = np.squeeze(pred)
        pred = pred.tolist()
        fenlei = ''
        index = pred.index(max(pred))
        '''获取最大值进行判断'''
        if index == 0:
            fenlei = '草体'

        elif index == 2:
            fenlei = '篆体'

        elif index == 3:
            fenlei = '隶书'

        else:
            fenlei = ''
        return render_template('upload.html', result=fenlei, imgpath=upload_path)
    return render_template('upload.html')
def kairec(upload_path):
        data = Image.open(upload_path)
        '''跟文件名没有关系'''
        pred = predict(5,'kaimodels/vgg.ckpt',data)
        pred = np.squeeze(pred)
        pred = pred.tolist()
        fenlei = ''
        index = pred.index(max(pred))
        '''获取最大值进行判断'''
        if index == 0:
            fenlei="楷体（"+"欧体"+"）"
        elif index == 1:
            fenlei = "楷体（" + "赵体" + "）"
        elif index == 2:
            fenlei = "楷体（" + "柳体" + "）"
        elif index == 3:
            fenlei = "楷体（" + "魏碑" + "）"
        elif index == 4:
            fenlei = "楷体（" + "颜体" + "）"
        return fenlei


if __name__ == '__main__':

    app.run(debug=True)