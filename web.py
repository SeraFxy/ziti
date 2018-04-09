# coding:utf-8

from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from nnets.vgg import vgg
import numpy as np
from PIL import Image, ImageFont, ImageDraw


app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 不使用GPU
inputs = tf.placeholder(tf.float32, shape=[None, None, 3])
example = tf.cast(tf.image.resize_images(inputs, [128, 128]), tf.uint8)
example = tf.image.per_image_standardization(example)
example = tf.expand_dims(example, 0)
output = vgg(example, 4, 1.0)
sess = tf.Session()
tf.train.Saver().restore(sess, 'models/vgg.ckpt')

'''导入模型'''
@app.route('/')
def about():
    return redirect(url_for('upload'))
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath,'static/uploads',f.filename)  #注意：没有的文件夹一定要先创建，不然会提示没有该路径

        f.save(upload_path)

        real_path = f.filename
        data = Image.open(upload_path)


        # f1 = open(upload_path)
        # real_path = os.path.realpath(f1.name)

        '''跟文件名没有关系'''
        pred = sess.run(output, feed_dict={inputs: data})
        pred = np.squeeze(pred)
        pred = pred.tolist()
        fenlei = ''
        index = pred.index(max(pred))
        '''获取最大值进行判断'''
        if index == 0:
            fenlei = u'草体'

        elif index == 2:
            fenlei = u'篆体'

        elif index == 3:
            fenlei = u'隶书'
        else:
            fenlei = u'楷体'
        return render_template('upload.html', result=fenlei,imgpath = real_path)
    return render_template('upload.html')



if __name__ == '__main__':

    app.run('0.0.0.0',port=5002,threaded=True)
