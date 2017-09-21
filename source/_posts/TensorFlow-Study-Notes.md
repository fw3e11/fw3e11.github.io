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

那么什么是`Operation`？就是**操作**。 我们从一个简单的例子入手：

``` python
#!/usr/bin/env python3

import tensorflow as tf

# Constants are initialized when you call tf.constant,
# and their value can never change.
node1 = tf.constant(3)  # Tensor("Const:0", shape=(), dtype=int32)
node2 = tf.constant(4.0)  # Tensor("Const_1:0", shape=(), dtype=float32)

print(node1)
print(node2)

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