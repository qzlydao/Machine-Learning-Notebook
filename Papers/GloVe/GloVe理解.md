[TOC]

## 1. 背景介绍

词向量在很多应用中都有重要作用。现今学习词向量的方法主要分两类：1）Global Matrix Factorization Methods和2）Local context window methods，它们各有优缺点，下面分别进行说明。

### 1.1 Matrix Factorization Methods（矩阵分解方法）

这类方法的主要思想是：对于一个描述语料库统计信息的大型矩阵，对其进行分解，用低秩近似生成词向量。



比较有代表的算法有：LSA、HAL(Hyperspace Analogue to Language)、COALS、Hellinger PCA(HPCA)



这类算法的优点是：

1. 训练速度快；
2. 充分利用了语料库的统计信息



但也有缺点：

1. 比如LSA，它在词类比任务上表现较差；
2. 比如HAL，在计算词的相似性时，一些高频词，比如the、and和其它词同时出现，它们对相似计算有较大影响，但实际上，它们和其它词在语义上没多大联系。



### 1.2 Shallow Window-Based Methods

这类方法主要通过预测窗口内的上下文词来学习词向量。



基于这种思想的算法有：

1. Bengio et al.(2003)

   应该是最早提出词向量的学者，他们用一个简单的神经网络来学习词向量。

2. Collobert and Weston(2008)

   他们将词向量的学习和下游任务分离，从而可以使用单词完整的上下文学习词向量。

3. Mikolov et al(2013a)提出的CBOW和Skip-Gram

   基于向量内积的单层神经网络

4. vLBL和ivLBL:



这些模型学得词向量能很好通过词类比任务(word analogy task)，但也有明显缺点：

它们不是直接使用语料库的共现统计，而是滑动窗口扫描上下文词，因此计算书无法利用重复数据。



## 2. GloVe Model

### 2.1 GloVe模型介绍

在词向量的无监督学习中，语料库中的词频统计信息是非常重要的信息。GloVe模型就用到了<font color=red>基于窗口的词共现矩阵。</font>

假设语料库中有下面三句话：

- I like deep learning.
- I like NLP.
- I enjoy flying.

通过滑动窗口，遍历一遍语料库，可以得到如下共生矩阵：

|  counts  |  I   | like | enjoy | deep | learning | NLP  | flying |  .   |
| :------: | :--: | :--: | :---: | :--: | :------: | :--: | :----: | :--: |
|    I     |  0   |  2   |   1   |  0   |    0     |  0   |   0    |  0   |
|   like   |  2   |  0   |   0   |  1   |    0     |  1   |   0    |  0   |
|  enjoy   |  1   |  0   |   0   |  0   |    0     |  0   |   1    |  0   |
|   deep   |  0   |  1   |   0   |  0   |    1     |  0   |   0    |  0   |
| learning |  0   |  0   |   0   |  1   |    0     |  0   |   0    |  1   |
|   NLP    |  0   |  1   |   0   |  0   |    0     |  0   |   0    |  1   |
|  flying  |  0   |  0   |   1   |  0   |    0     |  0   |   0    |  1   |
|    .     |  0   |  0   |   0   |  0   |    1     |  1   |   1    |  0   |



符号定义：

$X$：共生矩阵；

$X_{ij}$：词条$j$出现在词条$i$的上下文中的次数；

$X_{i}=\sum_{k}X_{ik}$：任意词出现在词条$i$上下文中的次数；

$P_{ij}=P(j|i)=\frac{X_{ij}}{X_{i}}$：词$j$出现在词$i$上下文中的概率。



作者从预料库中对目标词*ice*和*stream*的上下文词出现概率和比率进行了统计，统计结果如下表：

<img src="Co-occurrence probabilities.png" alt="Co-occurrence probabilities" style="zoom:50%;" />

通过分析可以发现：

1. 若$P_{ik}/P_{jk}$越大，则说明词$k$与词$i$在语义上比词$j$更接近；
2. 若$P_{ik}/P_{jk}$越小，则说明词$k$与词$j$在语义上比词$i$更接近；
3. 若$P_{ik}/P_{jk} \approx 1$，则说明词$k$在语义上与词$i$和$j$要么都接近，要么都相差很远。



既然共现概率比能体现词在语义上的相似程度，作者希望构造一个函数，输入为词向量$w_{i},w_{j},w_{k}$，输出为共现概率比$P_{ik}/P_{jk}$，那么应该有：
$$
F(w_{i},w_{j},\tilde{w}_{k})=\frac{P_{ik}}{P_{jk}} \tag{1}
$$
因为向量空间是线性的，那么$F$可以定义在两个向量差上：
$$
F(w_{i}-w_{j},\tilde{w}_{k})=\frac{P_{ik}}{P_{jk}} \tag{2}
$$
等式（2）右边是个标量，而$F$可能是一个复杂函数，$F$可改写为：
$$
F\left ( (w_{i}-w_{j})^{T}\tilde{w}_{k} \right )=\frac{P_{ik}}{P_{jk}} \tag{3}
$$
在词条共现矩阵中，目标词和上下文词是任意的，也就是说可以交换的：$w\leftrightarrow \tilde{w}、X \leftrightarrow X^{T}$。因此所求模型也应该是关于参数对称的，这里分两步求解。首先要求$F$在群$(\mathbb{R},+)$和$(\mathbb{R_{>0},\times})$之间是同态的：
$$
F\left( (w_{i}-w_{j})^{T} \tilde{w}_{k} \right)=\frac{F(w_{i}^{T}\tilde{w}_{k})}{F(w_{j}^{T}\tilde{w}_{k})} \tag{4}
$$
带入（3）式，有：
$$
F(w_{i}^{T}\tilde{w}_{k})=P_{ik}=\frac{X_{ik}}{X_{i}} \tag{5}
$$
(4)式的解为$F=\text{exp}$，带入(5)式有：
$$
w_{i}^{T}\tilde{w}_{k}=\text{log}(P_{ik})=\text{log}(X_{ik})-\text{log}(X_{i}) \tag{6}
$$
上式等式左边满足交换，但右边不满足，可以将$\text{log}(X_{i})$吸收到偏置项$b_{i}$中，同时添加$\tilde{w}_{k}$的偏置项$\tilde{b}_{k}$使等式满足对称性：
$$
w_{i}^{T}\tilde{w}_{k}+b_{i}+\tilde{b}_{k}=\text{log}(X_{ik}) \tag{7}
$$
但(7)是病态的，因为$X_{ik}$可能会是0。

其中一种做法是在对数项中添加偏移解决，即$\text{log}(X_{ik}) \rightarrow \text{log}(1+X_{ik})$，这种做法的主要缺点是：给共现矩阵中的每一项权值都相同，而我们知道共现矩阵中值很小的两个词其实在语义上没多大联系。

作者的做法是使用加权的最小二乘法作为损失函数：
$$
J=\sum_{i,j=1}^{V}f\left(X_{ij}\right)\left( w_{i}^{T}\tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\text{log}(X_{ij}) \right)^{2} \tag{8}
$$
$V$是字典大小，权重函数$f$需满足下面的条件：

1. $f(0)=0$. 同时当$x$趋近于0时，$f(x)$能快速收敛到0. 即$\lim_{x\rightarrow 0}f(x)\text{log}^{2}x$是有限的；
2. $f(x)$是非减函数，这样能保证很小的$X_{ij}$不会被过度加权；
3. $f(x)$在$x$取很大值时，$f(x)$也不会太大。



作者找到一个比较理想的$f(x)$:
$$
f(x)=\begin{cases} (x/x_{max})^{\alpha} \space \text{if} \space  x<x_{max} 
\\ 1 \space\space\space\space\space\space\space\space\space\space\space\space\space\space\space\space\space \text{otherwise}\end{cases} \tag{9}
$$
本文中，$x_{max}=100,\alpha=3/4$.



### 2.2 与其它模型的关系





### 2.3 模型复杂度分析



## 3. 实验及结果

作者分别在词类比任务、词相似度任务和命名体识别(NER)任务上对模型进行了评估。



### 3.1 Word analogies

word analogy分两种，一种是语义推理，比如：“Athens is to Greece as Berlin is to \___?”；另一种是语法推理，比如：“a is to b as c is to \_\_?”.

实验结果见下表：

<img src="Results on the word analogy task.png" alt="Results on the word analogy task" style="zoom:50%;" />

作者对实验结果做了如下总结：

1. GloVe模型的实验结果要比其他模型好；
2. 作者使用了word2vec模型，它的效果也要比之前的模型更好，作者分析是用了负采样方法，因为负采样方法要比层序softmax更好，同时和负采样的个数和语料库的选择也有关；
3. GloVe模型在很大的语料库上也能轻松训练，同时语料库越大，模型的效果也会更好；
4. 但更大的语料库，并不会让SVD方法得到的结果更好，这也从侧面证明了我们的模型使用加权的必要性。



### 3.2 Word similarity

在5个数据集上做了对比实验，结果如下：

<img src="word similarity test result.png" alt="word similarity test result" style="zoom:50%;" />

实验结果表明：GloVe模型得到的词向量在word similarity任务上也是表现最好的。CBOW表现也不错，但是CBOW用到的语料库是GloVe的两倍多一点。



### 3.3 Named entity recognition

作者在ConLL-03训练数据集上训练模型，在 1) ConLL-03 testing data, 2) ACE Phase 2 (2001-02) and ACE-2003 data,  3) MUC7 Formal Run test set上进行测试

<img src="F1 score on NER task.png" alt="F1 score on NER task" style="zoom:50%;" />

作者训练了一个CRF模型用于NER任务。实验结果表名，除了在CoNLL测试集上，HPCA模型表现更好外，在其它数据集上， 都是GloVe模型表现更好。因此作者认为GloVe词向量在NLP的下游任务中都是很有用的。

## 4. 模型分析

### 4.1 Vector Length and Context Size

<img src="analysis on vector length and window size.png" alt="analysis on vector length and window size" style="zoom:50%;" />

名词解释：

1. Symmetric context：能扩展到目标词左右词的窗口为Symmetric context；
2. Asymmetric context：只扩展到左边的上下文窗口称为非对称窗口。



实验结果分析：

1. 通过图(a)可以发现，当词向量长度超过200维后，维度的提升并不会带来结果的显著变化；

2. 通过图(b)(c)可以发现，在窗口比较小或使用非对称滑动窗口时，语法类任务的结果要比语义类任务结果好。

   从直观上理解，可能语法推理对紧挨着的词即词的顺序比较敏感，而语义推理的任务需要更大的滑动窗口，从而获取更多的信息。



### 4.2 Corpus Size

作者在不同的预料库上训练了300维的词向量用于词类比任务，其对比结果如下：

<img src="Analysis on corpus size.png" alt="Analysis on corpus size" style="zoom:50%;" />

通过实验结果可以发现：

1. 在语法类比任务(Syntactic analogy task)上，随着语料库规模的增加，准确率呈单调上升趋势；

2. 但是语料库规模的增加，语义类比任务(Semantic analogy task)的准确率并不是单调上升的。

   比如在Wiki2010和Wiki2014语料库上训练的词向量表现要比更大的语料库Gigaword5更好！

   作者分析原因如下：

   1. 测试数据集里包含很多地名和国家名的类比，而Wikipedia可能包含更多的可以理解这些地名和国家名的文档；
   2. Wikipedia会吸收新知识，而Giagaword库中都是不变的新闻，可能会有过时的或错的信息。



### 4.3 Run-time

整个训练过程时间开销分两部分：统计得到共现矩阵$X$和模型训练。

作者分别作了GloVe和基于负采样的CBOW和Skip-Gram模型的对照试验，结果如下：

<img src="Training time contrast on GloVe CBOW and Skip-Gram.png" alt="Training time contrast on GloVe CBOW and Skip-Gram" style="zoom:50%;" />

试验结果分析：

1. 作者在前文提到，训练300维以下的词向量，迭代50次停止，从试验结果看，基本上迭代到20次之后，效果提升并不明显；
2. Skip-Gramp模型比CBOW要好；
3. CBOW模型负样本越多，效果反而越差。（==why?==）



### 4.4 与word2vec的比较

精确定量地比较GloVe和word2vec很难，因为影响模型的参数有很多。前面已经从向量长度、窗口大小和语料库大小方面做了比较。

对于word2vec模型随着负样本的增加，表现反而不好的原因，作者估计是负样本并没有很好近似目标概率分布。



## 5. 结论

对于分布式词向量的学习，很多人都在关注到底是基于计数统计的方法(count-based methods)好，还是基于预测的方法更好(prediction-based methods)。作者通过本文所做的工作认为：两种方法本质上不会有太大的差别！因为它们都利用了词的共现统计信息。

但是，作者认为因为GloVe用到的是基于全局的统计，所以效果会更好。



