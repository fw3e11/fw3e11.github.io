---
title: TensorFlow Study Notes
date: 2017-09-20 13:22:44
tags:
  - Machine Learning
---

在国内安装TensorFlow，推荐使用中科大的镜像

``` bash
pip3 install tensorflow -i https://mirrors.ustc.edu.cn/pypi/web/simple
```

## Basic Concept
首先是几个概念：什么是Tensor，什么是Flow？
数据（Data）是怎么表示的？数据是如何处理（manipulate）的？

### Tensors
> the central unit of data in TensorFlow

`Tensor`就是数据，所以`TensorFlow`就是数据流转，通过数据的流转完成计算任务。是什么让数据流转起来？是**图**。

### The Computational Graph
> You might think of TensorFlow Core programs as consisting of two discrete sections:
> 1. Building the computational graph
> 2. Running the computational graph

`TensorFlow`可以理解为一个框架，让`Tensor` _Flow_ 起来就是
1. 构建一个“可计算的图”
2. 运行这个图

一切“运行”（`run`）的操作，必须在一个`Session`里面完成，`Session`包含了`TensorFlow`运行环境和执行上下文。
> A computational graph is a series of TensorFlow operations arranged into a graph of nodes.

我们可以看出，`Tensor`和`Node`和`Data`是基本等效的同一个概念。

那么什么是`Operation`？就是**操作**。 我们从一个简单的例子入手：

``` python
#!/usr/bin/env python3

import tensorflow as tf

# Constants are initialized when you call tf.constant,
# and their value can never change.
node1 = tf.constant(3)  # Tensor("Const:0", shape=(), dtype=int32)
node2 = tf.constant(4.0)  # Tensor("Const_1:0", shape=(), dtype=float32)

# Model parameters
# Variables are not initialized when you call tf.Variable.
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)   

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Loss
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(linear_model - y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer() # initial all variables

def main():
    with tf.Session() as session:
        print(session.run([node1, node2]))  # [3, 4.0]

        session.run(init)
        print(session.run(loss, {x: x_train, y: y_train}))  # 23.66

        fixW = tf.assign(W, [-1.])
        fixb = tf.assign(b, [1.])
        session.run([fixW, fixb])  # fixW, fixb are operations
        print(session.run(loss, {x: x_train, y: y_train}))  # 0.0

        session.run(init)  # reset variables

        for _ in range(1000):
            session.run(train, {x: x_train, y: y_train})

        print(session.run([W, b, loss], {x: x_train, y: y_train}))
        # [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32), 5.6999738e-11]


if __name__ == '__main__':
    main()
```

### tf.estimator
> `tf.estimator` is a high-level TensorFlow library that simplifies the mechanics of machine learning:
> * running the train loops
> * running the evaluation loops
> * managing data sets

`tf.estimator` 提供了相当丰富的模型，同时支持底层API构建出自定义的模型。具体代码请看[官网](https://www.tensorflow.org/get_started/get_started)
``` python
#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

class Tutorial(object):
    # Constants are initialized when you call tf.constant,
    # and their value can never change.
    node1 = tf.constant(3)  # Tensor("Const:0", shape=(), dtype=int32)
    node2 = tf.constant(4.0)  # Tensor("Const_1:0", shape=(), dtype=float32)

    # Model parameters
    # Variables are not initialized when you call tf.Variable.
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)

    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    # Loss
    y = tf.placeholder(tf.float32)
    loss = tf.reduce_sum(tf.square(linear_model - y))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # Training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]

    init = tf.global_variables_initializer() # initial all variables

    def hello_main(self):
        with tf.Session() as session:
            print(session.run([self.node1, self.node2]))  # [3, 4.0]

            session.run(self.init)
            print(session.run(self.loss, {self.x: self.x_train, self.y: self.y_train}))  # 23.66

            fixW = tf.assign(self.W, [-1.])
            fixb = tf.assign(self.b, [1.])
            session.run([fixW, fixb])  # fixW, fixb are operations
            print(session.run(self.loss, {self.x: self.x_train, self.y: self.y_train}))  # 0.0

            session.run(self.init)  # reset variables

            for _ in range(1000):
                session.run(self.train, {self.x: self.x_train, self.y: self.y_train})

            print(session.run([self.W, self.b, self.loss], {self.x: self.x_train, self.y: self.y_train}))
            # [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32), 5.6999738e-11]

    def estimator_main(self):
        feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
        estimator = tf.estimator.LinearRegressor(feature_columns)

        x_train = np.array(self.x_train)
        y_train = np.array(self.y_train)

        input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
        train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
        estimator.train(input_fn, steps=1000)
        print("estimator_main: ")
        print(estimator.evaluate(train_input_fn))

    def custom_estimator_main(self):
        estimator = tf.estimator.Estimator(model_fn=self._model_fn)
        x_train = np.array([1., 2., 3., 4.])
        y_train = np.array([0., -1., -2., -3.])
        input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
        train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
        estimator.train(input_fn=input_fn, steps=1000)
        print("custom_estimator_main: ")
        print(estimator.evaluate(train_input_fn))

    def _model_fn(self, features, labels, mode):
        # Build a linear model and predict values
        W = tf.get_variable("W", [1], dtype=tf.float64)
        b = tf.get_variable("b", [1], dtype=tf.float64)
        y = W * features['x'] + b
        # Loss sub-graph
        loss = tf.reduce_sum(tf.square(y - labels))
        # Training sub-graph
        global_step = tf.train.get_global_step()
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
        # EstimatorSpec connects subgraphs we built to the
        # appropriate functionality.
        return tf.estimator.EstimatorSpec(mode=mode, predictions=y, loss=loss, train_op=train)

if __name__ == '__main__':
    tutorial = Tutorial()
    tutorial.hello_main()
    tutorial.estimator_main()
    tutorial.custom_estimator_main()
```

## Softmax for MNIST
`MNIST` 可以称得上是机器学习领域的 _Hello world_。根据UFLDL中[Softmax Regression](http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression)的解释，`Softmax回归`是`逻辑回归`（二分类问题）在多分类问题上的推广。如果待分类的类别之间互斥，应该使用Softmax回归分类器，反之则应该建立K个逻辑回归分类器。
``` python
#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Datasets(train=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x1074f3630>,
# validation=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x107530588>,
# test=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x107530438>)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Tensor("Images:0", shape=(?, 784), dtype=float32)
# Tensor("Labels:0", shape=(?, 10), dtype=float32)
x = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28], name="Images")
labels = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="Labels")

# W and b are matrix initialized with zeros, when run tf.global_variables_initializer()
W = tf.Variable(tf.zeros([28 * 28, 10]), name="Weights")
b = tf.Variable(tf.zeros([10]), name="bias")

# softmax model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross entropy & train step
cross_entropy = -tf.reduce_sum(labels * tf.log(y), name="cross_entropy")
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy, name="train_step")

# accuracy
# tf.argmax returns the index of the highest entry in a tensor along some axis
# tf.cast casts a list of booleans to a list of float32 numbers
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        session.run(train_step, feed_dict={x: batch_x, labels: batch_y})
    print(accuracy.eval(feed_dict={x: mnist.test.images, labels: mnist.test.labels}))
```

## CNN for MNIST
不要满足于Softmax分类器达到的91%的准确率！我们使用一个稍微复杂一点的模型：卷积神经网络，来改善效果，以达到99%的准确率！
``` python
#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from stopwatch import StopWatch

class MnistTutorial(object):
    _x = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28], name="Images")
    _y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="Labels")
    _x_image = tf.reshape(_x, [-1, 28, 28, 1])
    _keep_prob = tf.placeholder(dtype=tf.float32)

    def __init__(self):
        self._train_step = None
        self._accuracy = None
        self._runnable = False
        self._mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def cnn(self, hidden_layers, fc_layer):
        h_pool = None
        last_layer = 1
        for layer in hidden_layers:
            W_conv = self._weight_variable([5, 5, last_layer, layer])
            b_conv = self._bias_variable([layer])
            x_conv = self._x_image if h_pool is None else h_pool
            h_conv = tf.nn.relu(self._conv2d(x_conv, W_conv) + b_conv)
            h_pool = self._max_pool_2x2(h_conv)
            last_layer = layer

        fc_nodes = int((28 / 2 / len(hidden_layers)) ** 2 * last_layer)
        W_fc = self._weight_variable([fc_nodes, fc_layer])
        b_fc = self._bias_variable([fc_layer])
        hidden_pool_flat = tf.reshape(h_pool, [-1, fc_nodes])
        h_fc = tf.nn.relu(tf.matmul(hidden_pool_flat, W_fc) + b_fc)
        h_fc_drop = tf.nn.dropout(h_fc, self._keep_prob)

        W_out = self._weight_variable([fc_layer, 10])
        b_out = self._bias_variable([10])
        y = tf.nn.softmax(tf.matmul(h_fc_drop, W_out) + b_out)

        cross_entropy = - tf.reduce_sum(self._y * tf.log(y))
        correct_prediction = tf.equal(tf.argmax(self._y, 1), tf.argmax(y, 1))
        self._train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                dtype=tf.float32))
        self._runnable = True
        return self

    def run(self, batch=50):
        if self._runnable is False:
            raise Exception("build cnn first")
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            stop_watch = StopWatch()
            for i in range(10000):
                batch_x, batch_y = self._mnist.train.next_batch(batch)
                if i % 100 == 0:
                    train_accuracy = session.run(self._accuracy, feed_dict={
                        self._x: batch_x, self._y: batch_y, self._keep_prob: 1
                    })
                    print("step %d: training accuracy %g using %dms"
                          % (i, train_accuracy, stop_watch.elapsed()))
                self._train_step.run(feed_dict={self._x: batch_x,
                                                self._y: batch_y,
                                                self._keep_prob: 0.5})
            print("test accuracy: %g" %
                  self._accuracy.eval(feed_dict={
                      self._x: self._mnist.test.images,
                      self._y: self._mnist.test.labels,
                      self._keep_prob: 1
                  }))

    @staticmethod
    def _weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

    @staticmethod
    def _bias_variable(shape):
        return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32))

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME')

# test accuracy: 0.9919
```
