

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data

"""An Example of a DNNClassifier for the Iris dataset."""
"""
本案例是tensorflow 官方文档之鸢尾花品种预测案例.
参考文档: https://www.tensorflow.org/get_started/get_started_for_beginners
"""

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

# 官方源码,采用CNN网络预测
def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)

    # Train the Model.
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 5.9],
        'SepalWidth': [3.3, 3.0, 3.0],
        'PetalLength': [1.7, 4.2, 4.2],
        'PetalWidth': [0.5, 1.5, 1.5],
    }

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


    # tensorflow提供了多种写日志文件的API
    sess = tf.Session()
    writer = tf.summary.FileWriter('./log', sess.graph)
    writer.close()


# 采用线性规划的方式预测
def test():
    # Build Model
    x = tf.placeholder(tf.float32, [None, 4])
    y_label = tf.placeholder(tf.float32, [None, 3])
    w = tf.Variable(tf.zeros([4, 3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.nn.softmax(tf.matmul(x, w)+b)

    # Loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), reduction_indices=[1]))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # Prediction
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_label, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # tensorflow提供了多种写日志文件的API
        writer = tf.summary.FileWriter('./log', tf.get_default_graph())
        writer.close()
        for step in range(1001):
            batch_x = train_x
            batch_y = train_y
            sess.run(train, feed_dict={x: batch_x, y:batch_y})


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    # tf.app.run(test())

# 总算弄明白了,这个案例和MNIST案例不同之处在于
# 训练集中用的 (x,y) , 测试集用的(testx, testy), 预测集用的(pre_x, pre_y)
# 训练集和测试集用的不同的数据, 可能对结果的判断更准确些

