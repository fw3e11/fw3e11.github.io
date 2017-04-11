---
title: 'Spark ML: Kaggle Titanic Solution'
date: 2017-04-11 15:21:36
tags:
  - Java
  - Spark
  - Machine Learning
---

因为平时工作的主要语言是Java，在接触机器学习的时候，首先想到目前较火的`Spark`和`Spark-MLlib`。现在的版本已经升级到了`2.1.0`，然而在网上的文章主要是`Python`，`Scala`和`Java Spark 1.6`，经过了一个周末的摸索，终于学会了做一个调包侠。
``` xml
<dependency>
  <groupId>org.apache.spark</groupId>
  <artifactId>spark-core_2.10</artifactId>
  <version>2.1.0</version>
</dependency>
<dependency>
  <groupId>org.apache.spark</groupId>
  <artifactId>spark-mllib_2.10</artifactId>
  <version>2.1.0</version>
</dependency>
```
### Spark: Hello world
首先是国际惯例，Hello world。这一段主要是参考《Spark 机器学习》（Machine Learning with Spark）这本书。首先是数据：

``` text
John,iPhone Cover,9.99
John,Headphones,5.49
Jack,iPhone Cover,9.99
Jill,Samsung Gallaxy Cover,8.95
Bob,iPad Cover,5.49
```

JavaSparkContext 环境
``` java
JavaSparkContext sparkContext = new JavaSparkContext("local[*]", "First Java Spark App");
sparkContext.setLogLevel("WARN");
```

前三个例子都没什么问题
``` java
JavaRDD<String[]> data = sparkContext.textFile("data.txt")
                .map((String string) -> string.split(","));
System.out.println("Total purchases: " + data.count());
System.out.println("Unique users: " + data
                .map((String[] strings) -> strings[0])
                .distinct().count());
System.out.println("Total revenue : " + data
                .map((String[] strings) -> Double.parseDouble(strings[2]))
                .reduce((Double v1, Double v2) -> v1 + v2));
```

最后一个求畅销品的例子有一些问题：
1. `JavaRDD`没有`reduceByKey()`这个方法。应该把`JavaRDD`转化为`JavaPairRDD`，这里不能用`map()`方法，取而代之的是使用`mapToPair()`方法。
2. 得到类型为`List<Tuple2<String, Integer>>`的`pairs`后，对`pairs`直接使用`sort()`方法会报错`java.lang.UnsupportedOperationException`。这里的问题是`pairs`的运行时类型是`scala.collection.convert.Wrappers$SeqWrapper`。我的解决办法比较low：`pairs = new ArrayList<>(pairs);`

``` java
List<Tuple2<String, Integer>> pairs = data
                .mapToPair((String[] strings) -> new Tuple2<>(strings[1], 1))
                .reduceByKey((Integer v1, Integer v2) -> v1 + v2)
                .collect();
pairs = new ArrayList<>(pairs);
pairs.sort(Comparator.comparingInt(o -> -o._2));
System.out.println(String.format("Most popular product: %s with %d purchases", pairs.get(0)._1(), pairs.get(0)._2()));
```

到此我们就完成了Java Spark Hello world。

### Spark MLlib: Kaggle Titanic Disaster
随机森林什么的我还没有学到，就是跟着Andrew NG的视频学了逻辑回归，感觉用来解决是否幸存的二分类问题没毛病啊——就是TA了！

首先还是数据：
1. 不能简单的用`,`分隔：双引号引起来的Name
2. 字段有缺失：比如Age
3. 数据有没有意义？比如Ticket

``` text
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27,0,2,347742,11.1333,,S
10,1,2,"Nasser, Mrs. Nicholas (Adele Achem)",female,14,1,0,237736,30.0708,,C
11,1,3,"Sandstrom, Miss. Marguerite Rut",female,4,1,1,PP 9549,16.7,G6,S
12,1,1,"Bonnell, Miss. Elizabeth",female,58,0,0,113783,26.55,C103,S
13,0,3,"Saundercock, Mr. William Henry",male,20,0,0,A/5. 2151,8.05,,S
14,0,3,"Andersson, Mr. Anders Johan",male,39,1,5,347082,31.275,,S
15,0,3,"Vestrom, Miss. Hulda Amanda Adolfina",female,14,0,0,350406,7.8542,,S
16,1,2,"Hewlett, Mrs. (Mary D Kingcome) ",female,55,0,0,248706,16,,S
17,0,3,"Rice, Master. Eugene",male,2,4,1,382652,29.125,,Q
......
```

这里我走了许多弯路，不说了，说多了都是泪。其实是应该使用`DateFrameReader`。
具体代码如下：
``` java
SparkSession sparkSession = SparkSession.builder()
        .appName("Kaggle Titanic").master("local[*]")
        .getOrCreate();
sparkSession.sparkContext().setLogLevel("WARN");

Dataset<Row> trainingDataset = sparkSession.read()
                .option("header", true)
                .csv(TRAINING_FILEPATH)
                .filter(line -> line.get(5) != null) // 过滤掉没有Age信息的乘客
                .drop("Ticket")
                .drop("Cabin")
                .cache();
trainingDataset.printSchema();
```

打印的结果如下：
``` console
root
 |-- PassengerId: string (nullable = true)
 |-- Survived: string (nullable = true)
 |-- Pclass: string (nullable = true)
 |-- Name: string (nullable = true)
 |-- Sex: string (nullable = true)
 |-- Age: string (nullable = true)
 |-- SibSp: string (nullable = true)
 |-- Parch: string (nullable = true)
 |-- Fare: string (nullable = true)
 |-- Embarked: string (nullable = true)
```

而`LogisticRegression`的`train()`方法需要的是：
``` console
root
 |-- label: double (nullable = false)
 |-- features: vector (nullable = false)
```

这里使用`createDataFrame()`方法来构造一个符合`LogisticRegression`需求的`Dataset<Row>`：
``` java
List<Row> data = new ArrayList<>();
for (Row row : trainingDataset.collectAsList()) {
  double label = Double.parseDouble(row.getString(1));
  double pClass = Double.parseDouble(row.getString(2));
  double sex = "male".equals(row.getString(4)) ? 1.0d : 0d;
  double age = Double.parseDouble(row.getString(5));
  data.add(RowFactory.create(label, Vectors.dense(pClass, sex, age)));
}
sparkSession.log().debug("{}", data.size());
trainingDataset = sparkSession.createDataFrame(data,
  new StructType(new StructField[]{
    new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
    new StructField("features", new VectorUDT(), false, Metadata.empty()),
}));
```
这一块儿我们在数据的预处理上简单粗暴：缺失的直接抛弃；看不懂的数据一概不用；看过电影的都知道“妇女小孩优先”，和客舱等级（社会地位）也可能有关系，所以我们就使用这3个维度。

下面就是训练和预测，关键代码粘贴如下：
``` java
private static final LogisticRegression lr = new LogisticRegression();
private LogisticRegressionModel lrModel;

lrModel = lr.train(trainingDataset);

System.out.println("Coefficients:\n" + lrModel.coefficientMatrix());
```

打印结果是这样的：
``` console
Coefficients:
-1.2885462256648932  -2.5221360948672262  -0.03692891763134605  
```
分别是pClass, sex, age。（其中1是男性，0是女性）说明：
1. 随着客舱等级（数字）的上升=>社会等级下降=>获救概率下降；
2. 女士优先在灾难中时间的比较好；
3. 年龄大小（貌似）和是否获救无关；

当然更详细的分析由数据科学家和算法工程师来做吧，我接着做我的调包侠，下一步就是预测了：
``` java
public Double predict(Vector features) {
  return lrModel.predict(features);
}
```

有没有问题？当然有，在训练数据中features缺失我们可以视而不见，在测试数据中features缺失我们可不能不给结果啊。我的解决办法是用平均数代替：
``` java
double avgAge = trainingDataset.toJavaRDD()
                .mapToDouble(row -> Double.parseDouble(row.getString(5)))
                .mean();
```

分数很低，求轻喷
![complete](/img/complete.png)
