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

## 几个概念
首先是几个概念
### Tensors
> the central unit of data in TensorFlow

`Tensor`就是数据，所以`TensorFlow`就是数据流转，通过数据的流转完成计算任务。数据如何流转？**图**。

### The Computational Graph
> You might think of TensorFlow Core programs as consisting of two discrete sections:
> 1. Building the computational graph
> 2. Running the computational graph

`TensorFlow`可以理解为一个框架，让`Tensor`“Flow”起来就是
1. 构建一个“可计算的图”
2. 运行这个图

> A computational graph is a series of TensorFlow operations arranged into a graph of nodes.

我们可以看出，`Tensor`和`Node`和`Data`是基本等效的同一个概念。

那么什么是`Operation`？就是**操作**。

``` python
#!/usr/bin/env python3

import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(linear_model - y))

train_data = {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = tf.group(optimizer.minimize(loss))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(loss, train_data))

    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    session.run([fixW, fixb])  # fixW, fixb are operations
    print(session.run(loss, train_data))

    session.run(tf.global_variables_initializer())  # reset variables
    for _ in range(1000):
        session.run(train, train_data)

    print(session.run([W, b]))

```
