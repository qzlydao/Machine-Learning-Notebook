# Distributed Representations of Sentences and Documents理解

作者：Quoc Le, Tomas Mikolov

于2014年发表于*Proceedings of the 31^st^ International Conference on Machine Learning*



BOW(bag-of-word)的两个主要缺点：

1. 没有考虑词的顺序
2. 忽略了词与词之间的语义关系；



本文通过一种==无监督?==的学习，从文本中学习单词的定长特征向量。



## 1. Introduction

以前常用的word embedding有bow或bag-of-n-grams，但存在如下缺点：

1. 没有考虑词序；
2. 数据稀疏，维度太高（一般词表单词是非常多的）；
3. 无法考虑词与词之间的相似性（比如，powerful和strong）



## 2. 算法

作者回顾了几种学习词向量的算法。



### 2.1 Learning Vector Representation of Words

这一节主要介绍词的分布式向量表示。



下图是一个常用的学习词向量的框架：

<img src="/Users/liuqiang/Documents/Machine_Learning/我的文章/word2vec/A framework for learning word vectors.jpg" alt="A framework for learning word vectors" style="zoom:50%;" />



给定一个单词序列（可以理解为一句话）$w_{1},w_{2}, \cdots, w_{T}$，其Object Function可表示为最大化平均对数似然函数：

<img src="Object Function.jpg" alt="Object Function" style="zoom:50%;" />

这里，条件概率$p(w_{t}|w_{t-k}, \cdots, w_{t+k})$是一个多分类问题，可用softmax函数表示：

<img src="softmax function.jpg" alt="softmax function" style="zoom:50%;" />

这里，$y_{i}$是目标词的非正则化的对数概率，计算公式如下：
$$
y = b + Uh(w_{t-k}, \cdots, w_{t+k};W) \tag{1}
$$
其中$U,b$是softmax参数，$h$是原始输入向量（从$W$中抽取的）的平均或堆叠。

==在实践中，为了加速训练，更多的是使用hierarchical softmax，而不是softmax==，作者做了些优化，用Huffman Tree来构造hierarchical  softmax。



在具体训练词向量时，作者使用的是神经网络模型。



### 2.2 Paragraph Vector: A distributed memory model

思路和训练词词向量一样：

<img src="A framework for learning paragraph vector.jpg" alt="A framework for learning paragraph vector" style="zoom:50%;" />

































