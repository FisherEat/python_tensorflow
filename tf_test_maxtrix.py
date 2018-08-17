'''
经过一个月的停滞,再次重拾TensorFlow深度学习框架.
本案例模拟TensorFlow案例
该案例为介绍: Tensor Variable Feed 和 Fetch
'''

import tensorflow as tf
import numpy as np

# assign 函数测试
state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# tensor 运算测试
ts1 = tf.constant(1.0, shape=[2, 2])
ts2 = tf.Variable(tf.random_normal([2, 2]))

# 基本数学函数

# 以下x,y均代表tensor
input1 = tf.Variable(tf.random_uniform([3]))
input2 = tf.constant([1.0, 20., 40.0])
x = tf.constant(100.0)
y = tf.constant(2.0)
tf.add_n([input1, input2], name=None)  # inputs:tensor数组，所有tensor相加
tf.abs(x, name=None)         # 绝对值
tf.negative(x, name=None)    # 取反
tf.sign(x, name=None)        # 取符号(y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.)
tf.square(x, name=None)      # y=x*x
tf.round(x, name=None)       # Rounds the values of a tensor to the nearest integer, element-wise.
tf.sqrt(x, name=None)        # sqrt
tf.pow(x, y, name=None)      # x,y均为tensor，element-wise求pow
tf.exp(x, name=None)         # y=e^x
tf.log(x, name=None)         # y=log(x)
tf.ceil(x, name=None)        # ceil
tf.floor(x, name=None)       # floor
tf.maximum(x, y, name=None)  # z=max(x,y)
tf.minimum(x, y, name=None)
tf.cos(x, name=None)         # 三角函数,sin,cos,tan,acos,asin,atan
tf.sin(x, name=None)
tf.tan(x, name=None)
tf.acos(x, name=None)
tf.asin(x, name=None)
tf.atan(x, name=None)
#等等一些函数。

# matrix 测试
martrix1 = tf.constant([[3., 5.], [9., 9.], [4., 9.], [8., 8.]])
martrix2 = tf.constant([[2., 7.], [5., 9.]])
martrix3 = tf.constant([[2.0, 3.0]])
product = tf.matmul(martrix1, martrix2)

mx1 = tf.diag([1, 1, 1, 1], name="diag") # 得到以diagonal为对角的tensor #
mx2 = tf.diag_part([[1, 2], [2, 3]], name="diag_part") # tf.diag 逆操作,得到input的对角矩阵
mx3 = tf.transpose([[1, 3], [2, 4]], perm=None, name=None) # 转置矩阵,y[i,j]=x[j,i]
#矩阵乘法
mx4 = tf.matmul(martrix2, martrix3,
  transpose_a=False, transpose_b=True,  #转置
  adjoint_a=False, adjoint_b=False,      #共轭
  a_is_sparse=False, b_is_sparse=False,  #矩阵是否稀疏
  name=None)

# Reduction 归约操作\降维操作
# （1）tf.reduce_sum
# 当keep_dims=False。rank of tensor会降维度。
tf.reduce_sum(mx1,
   axis=None,               # 要归约的dimention。值为None或一个数字或者数组。如0,1,[0,3,4]
   keep_dims=False,         # if true, retains reduced dimensions with length 1.
   name=None,
   reduction_indices=None)

# （2）tf.reduce_min / tf.reduce_max / tf.reduce_mean
# 参数与tf.reduce_sum一致。
# tf.reduce_min : 被归约的数取最小值；
# tf.reduce_max : 被归约的数取最大值；
# tf.reduce_mean: 被归约的数取平均值。

# （3）逻辑操作
# tf.reduce_all：logical and operation
# tf.reduce_any: logical or operation


# （4) 自定义操作函数
# tf.einsum(equation, *inputs)
#例子：
# tf.einsum('ij,jk->ik', ts1,ts2)  #矩阵乘法
# tf.einsum('ij->ji',ts1)          #矩阵转置

# 测试reduce_mean
A = np.array([[1, 2], [3, 4]])
reduce1 = tf.reduce_mean(A)
reduce2 = tf.reduce_mean(A, axis=0)
reduce3 = tf.reduce_mean(A, axis=1)
# 输出
# 2 #整体的平均值
# [2 3] #按列求得平均
# [1 3] #按照行求得平均

# tensor 大小比较
eq1 = [1, 2]
eq2 = [3, 4]
tf.equal(eq1, eq2, name=None)
tf.not_equal(eq1, eq2, name=None)
tf.less(eq1, eq2, name=None)
tf.less_equal(eq1, eq2, name=None)
tf.greater(eq1, eq2, name=None)
tf.greater_equal(eq1, eq2, name=None)

# 类型转换测试
# tf.cast(x, dtype, name=None)
#Casts a tensor to a new type.

#For example:
# tensor `a` is [1.8, 2.2], dtype=tf.float
#tf.cast(a, tf.int32) ==> [1, 2]  dtype=tf.int32

# 测试reshape
rank1 = tf.ones([3, 4, 5])
rank2 = tf.reshape(rank1, [6, 10])
rank3 = tf.reshape(rank1, [3, -1])

# https://blog.csdn.net/yjt1325/article/details/79758602
# https://wdxtub.com/2017/05/31/tensorflow-learning-note/
# 测试矩阵与向量相乘 & 相加


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # tensorflow提供了多种写日志文件的API
    writer = tf.summary.FileWriter('./log', sess.graph)
    writer.close()

    ts_add1 = tf.add(ts1, ts2, name="add")
    ts_sub1 = tf.subtract(ts1, ts2, name="subtract")
    ts_mul1 = tf.multiply(ts1, ts2, name="multiply")
    ts_div1 = tf.multiply(ts1, ts2, name="division")
    # tensor-scalar运算测试
    ts_add2 = ts1 + 2
    ts_sub2 = ts1 - 2
    ts_mul2 = ts1 * 2
    ts_div2 = ts1 / 2
    print(sess.run(ts_add1))

    # 测试diag
    print(sess.run(mx1))
    print(sess.run(mx2))
    print(sess.run(mx3))
    print("mx4 = : ", sess.run(mx4))

    # reshape
    print(sess.run(rank1))
    print(sess.run(rank2))
    print(sess.run(rank3))

    print(sess.run(martrix2 + martrix3))

    # 矩阵与向量相加 & 相乘
    # 参考:https://wdxtub.com/2017/05/31/tensorflow-learning-note/
    l1 = tf.constant([1.0, 1.0, 1.0, 1.0])
    l2 = tf.constant(12.0, shape=[4])
    output = tf.add(l1, l2)
    print(sess.run(output))
    print("matrix2  X matrix 3: ", sess.run(tf.matmul(martrix1, martrix2)))

    diag1 = tf.diag([1.0, 2.0, 3.0, 4.0])
    print(sess.run(diag1))

    x1 = tf.constant([[2., 7.], [5., 9.]])
    x2 = tf.constant([2.0, 3.0])
    print("矩阵和向量加法: ", sess.run(x1 + x2))

    g0 = tf.constant([2, 3])
    g1 = tf.constant([[2, 3]])
    g2 = tf.constant([[0, 1], [2, 3]])
    g3 = tf.matmul(g1, g2)
    mul = tf.reduce_sum(tf.multiply(g0, g2), reduction_indices=1)
    print("矩阵乘法", sess.run(g3))
    print("换一种方式是的矩阵相乘", sess.run(mul), sess.run(tf.multiply(g0, g2)))

    # print(sess.run(state))
    # for _ in range(3):
    #     sess.run(update)
    #     print(sess.run(state))

