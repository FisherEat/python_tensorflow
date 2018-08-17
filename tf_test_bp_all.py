import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow 全连接网络实现
# 参考 文档:https://blog.csdn.net/shine19930820/article/details/78359249

# 1. 准备数据, 这里用函数模拟数据
# x_data.shape = (300, 1), np.newaxis 将数据转化成 (300, 1)
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]

# 定义噪音波动, shape和x_data一样,也是 (300, 1), astype copye the array and cast to float32
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)

# y_data, 定义函数 y_data = 2*x_data^3 + x_data^2 + noise, y_data = (300, 1)
y_data = 2 * np.power(x_data, 3) + np.power(x_data, 2) + noise


# 2. 定义网络结构
# 定义占位符
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 定义两个隐藏层,一个输出层
# 隐藏层1 , relu(xs * Weights1 + biases1), relu((300, 1)* (1, 5) + (1, 5))
Weights1 = tf.Variable(tf.random_normal([1, 5]))
biases1 = tf.Variable(tf.zeros([1, 5]) + 0.1)
Wx_plus_b1 = tf.matmul(xs, Weights1) + biases1

# l1.shape= (300, 5)
l1 = tf.nn.relu(Wx_plus_b1)

# 隐藏层2, relu(l1 * Weights2 + biases2), relu((300, 5) * (5, 10) + (1, 10))
Weights2 = tf.Variable(tf.random_normal([5, 10]))
biases2 = tf.Variable(tf.zeros([1, 10]) + 0.1)
Wx_plus_b2 = tf.matmul(l1, Weights2) + biases2

# l2.shape = (300, 10)
l2 = tf.nn.relu(Wx_plus_b2)

# 输出层 prediction = (300, 10) * (10, 1) + (1, 1) = (300, 1)
Weights3 = tf.Variable(tf.random_normal([10, 1]))
biases3 = tf.Variable(tf.zeros([1, 1]) + 0.1)
prediction = tf.matmul(l2, Weights3) + biases3

# loss = MAE(y,ŷ )=(1/nsamples)∑i=1n (yi−yi^)2
# 定义loss表达式

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])
                      )
# optimizer 最小化loss,梯度下降法
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # 绘制原始x-y散点图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()

    # 训练10000次
    for i in range(10000):
        # 训练
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        # 每500步绘图并打印输出
        if i % 500 == 0:
            # 可视化模型输出的结果
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass

            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            loss_value = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
            print(prediction_value)
            print("\n")
            print(loss_value)
            # 绘制模型预测值, 预测的是y值, 在x_data共用的情况下,模型预测的是y值, 原来的scatter散点图和曲线图对比
            # 散点图为训练数据, 曲线图为在同一 x_data的前提下,y_prediction 的预测值., 预测结果与原来的散点图吻合.
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(1)

