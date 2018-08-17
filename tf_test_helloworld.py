import tensorflow as tf
import os
import iris_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hello = tf.constant('Hello, TensorFlow by gaolong!')
sess = tf.Session()
print(sess.run(hello))
print(iris_data.TRAIN_URL)