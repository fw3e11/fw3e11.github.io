---
title: Java Singleton Pattern
date: 2017-04-05 10:45:52
tags:
  - Java
---

### 饿汉式
这种方法简单明了，`static`和`final`关键字保证了instance变量在类第一次加载到内存时就会初始化，所以创建实例本身是线程安全的。
``` java
public class Singleton {
  private static final Singleton instance = new Singleton();
  private Singleton() {}
  public static Singleton getInstance() {
    return instance;
  }
}
```
问题也很明显，在类加载后立即初始化的方式不适用于运行时的参数依赖等需要Lazy Initialization等模式的情况。有时候必须`Lazy`——懒。

### 懒汉式
#### 线程不安全
教科书式的简单明了的代码，而且使用懒加载模式，却是线程不安全的：在多线程调用`getInstance()`时会创建多个实例。因此在多线程下不能正常工作。
``` java
public class Singleton {
  private static Singleton instance;
  private Singleton() {} // 私有构造器

  public static Singleton getInstance() {
    if (null == instance) {
      instance = new Singleton();
    }
    return instance;
  }
}
```

#### 线程安全 synchronized
为了解决懒加载模式线程不安全的问题，最简单的办法是将`getInstance()`方法加上一个关键字`synchronized`。
``` java
public static synchronized Singleton getInstance() {
  if (null == instance) {
    instance = new Singleton();
  }
  return instance;
}
```
虽然做到了线程安全，但是它并不高效。因为`synchronized`是排他的，任何时候只能有一个线程调用`getInstance()`方法，但是同步操作只在第一次调用时才需要。因此引入了双重检验锁（Double Checked Lock）

#### 线程安全 DCL
``` java
private static volatile Singleton instance; // volatile

public static Singleton getInstance() {
  if (null == instance) {             // 1st check
    synchronized (Singleton.class) {
      if (null == instance) {         // 2nd check
        instance = new Singleton();
      }
    }
  }
  return instance;
}
```
DCL的重点在于`volatile`这个关键字。因为`instance = new Singleton()`这个语句，虽然是一行，但并非一个原子操作。在`JVM`中大概做了以下3件事
1. 给`instance`分配内存
2. 调用构造函数来初始化成员变量
3. 将`instance`对象指向分配的内存空间（执行完这一步`instance`就非`null`了）

但在JVM的即时编译器中存在指定重排的优化。
在JVM的即时编译器中存在指定重排的优化。
JVM的即时编译器中存在指定重排的优化。

最终的执行顺序不保证是1-2-3，也可能是1-3-2。
在1-3-2的情况下，一个线程执行完3后被另一个线程抢占，此时instance已经为非null，所以线程二会直接返回instance。

`volatile` 有两个用途：
1. 保证线程内不会存有instance副本，每次读取主内存中的instance。
2. 进制指令重排优化，对于`volatile`变量的写操作`happen before`读操作。

### 内部静态类
《Effective Java》中推荐的方法。
``` java
public class Singleton {
  private static class SingletonHolder {
    private static final Singleton INSTANCE = new Singleton();
  }
  private Singleton() {}
  public static final Singleton getInstance() {
    return SingletonHolder.INSTANCE;
  }
}
```
1. 由JVM本身机制保证线程安全
2. Holder私有，同时也是懒汉式的
3. 读实例不会进行同步，没有性能缺陷
4. 不依赖JDK版本

### 枚举 Enum
``` java
public enum EasySingleton {
  INSTANCE;
}
```
