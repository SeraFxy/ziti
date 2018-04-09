import tensorflow as tf

with tf.Session() as sess:

    #初始化变量
    sess.run(tf.global_variables_initializer())

    #获取最新的checkpoint，其实就是解析了checkpoint文件
    latest_ckpt = tf.train.latest_checkpoint("./models")

    #加载图
    restore_saver = tf.train.import_meta_graph('./models/vgg.ckpt.meta')

    #恢复图，即将weights等参数加入图对应位置中
    restore_saver.restore(sess, latest_ckpt)

    #将图中的变量转为常量
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def , ["vgg"])
    #将新的图保存到"/pretrained/graph.pb"文件中
    tf.train.write_graph(output_graph_def, 'pretrained', "graph.pb", as_text=False)