import tensorflow as tf

# import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# 卷积神经网络会有很多的权重和偏置需要创建
# 定义初始化函数以便重复使用
# 权重函数是设置 初始卷积核
def weight_variable(shape):
    # 给权重制造一些随机的噪声来打破完全对称,例如这里截断的正态分布,标准差为0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 这个是加入噪音因子
def bias_variable(shape):
    # 由于使用Relu,也给偏置增加一些小的正值(0.1)用来避免死亡节点(dead neurous)
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层和池化层也是接下来重复使用的
# 定义卷积层和池化层
# x 为input, 即需要卷积的图片, W为卷积核
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 在正式设计卷积神经网络之前,先定义输入placeholder,x是特征,y是真实的label
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层
# 先使用前面写好的函数进行参数初始化,包括weigth和bias,
# 接着使用conv2d函数进行卷积操作,并加上偏置,
# 然后在使用ReLu激活函数进行非线性处理,
# 最后,使用最大池化函数对卷积的输出结果进行池化操作

# 注意: 因为采用的是 'SAME' 步长, 所以卷积后图片的尺寸不变,还是 28*28, 池化后尺寸改变
W_conv1 = weight_variable([5, 5, 1, 32]) # patch 5x5, in size 1, out size 32
# 矩阵用大写字母开头,便于自己下面区分(这只是个人建议)
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1) # output size 14x14x32


# 定义第二层卷积层
# 步骤如上一层,只是参数有所改变而已

# 注意: 因为采用的是 'SAME' 步长, 所以卷积后图片的尺寸不变,还是 14*14, 池化后尺寸改变
W_conv2 = weight_variable([5, 5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2) # output size 7x7x64

# 全连接层, 即隐藏层,这里只有一个隐藏层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减轻过拟合,增加一个dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 将dropout层的输出连接到一个softmax层,得到最后的概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
pre = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 定义loss函数和精度
cross_entropy = tf.reduce_mean(tf.reduce_sum(y*tf.log(pre), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义评测准确率的操作
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pre, 1))
accuray = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuray.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
    # 全部训练完后,在最终的测试集上进行全面的测试,得到整体分类准确率
    print("test accuracy %g" % accuray.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
