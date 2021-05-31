[TOC]

## 1. 论文概述

要解决的问题：

将维基百科作为唯一的知识源来回答开放域的问题，问题的答案是维基文档中的一个文本片段。



难点：

1. 文档检索，找到相关的文章
2. 机器理解，从相关文章中找到问题的答案



作者开发的基于维基百科的知识问答系统DrQA，主要包括下面两个模块：

1. Document Retriever，对于给定的问题，它可以高效的返回相关文章。主要用到的技术有倒排索引，bigram hash和TF-IDF;
2. Document Reader，一个多层的RNN机器理解模型，从上一步返回的文章中找答案。



## 2. 相关研究

**Open-Domain QA定义：**

> Open-domain QA was originally defined as finding answers in collections of unstructured documents, following the setting of the annual TREC competitions1 .



随着KBs的发展，诞生了一些基于KBs（作者提到的一些KBs介绍见[7.3 KBs](#7.3 KBs)）的数据集，比如：

- WebQuestions (Berant et al., 2013)
- SimpleQuestions (Bordes et al., 2015)

但是KBs有其局限性：

- 不完整（incompleteness）
- 主题固定（fixed schemas）



基于深度学习的机器理解：

- QuizBowl (Iyyer et al., 2014)
- CNN/Daily Mail based on news articles (Hermann et al., 2015)
- CBT based on children books (Hill et al., 2016)
- SQuAD (Rajpurkar et al., 2016) based on Wikipedia
- WikiReading (Hewlett et al., 2016) based on Wikipedia



其他人基于Wikipedia的QA方案介绍：

- Ryu et al. (2014)  方案：

  Wikipedia-based knowledge model + semi-structured knowledge

- Ahn et al. (2004) 

  Wikipedia + 其他资源

-  Buscaldi and Rosso (2006)

  用到了Wikipedia，但不是用着知识源，而是验证系统返回的答案。



其它full pipeline QA方案：

- QuASE (Sun et al., 2015)

  利用互联网作为知识源

- Microsoft’s AskMSR

  基于Wikipedia的，但它其实是个搜索引擎，关注点不在机器理解上

- IBM’s DeepQA

  基于Wikipedia这样的非结构文档，同时也会从结构化的数据库中搜答案

- YodaQA

  也是结合了网页、KBs、Wikipedia等

用Multitask learning实现多个问答数据集的结合，以期达到：

- 利用任务迁移来提高模型在数据集上的表现；
- 利用数据集的多样性提供一个能回答各种问题的工具

因此一些学者基于Multitask learning做了一些工作：

- Fader et al. (2014) 

  用了We- bQuestions, TREC，WikiAnswers和4个KBs

- Bordes et al. (2015) 

  结合了 WebQuestions，SimpleQuestions，Freebase

  但是在一个数据集上训练，再另外一个数据集上测试，效果就不行了，说明任务迁移（task transfer）确实是需面对的挑战。

作者介绍了这么多相关工作进行对比，主要是强调：

1. 自己的系统只依赖Wikipedia；
2. 自己的系统更专注机器对文本的理解；
3. 用了Multitask learning，在多个数据集上做了测试，是有普适性的。



## 3. DrQA介绍

这一节介绍作者开发的QA系统DrQA，系统概图：

<img src="An Over view of DrQA.jpg" alt="An Over view of DrQA" style="zoom:50%;" />

DrQA包括两大模块：文档检索(Document Retriever)和文档阅读(Document Reader)，下面详细介绍这两个模块：

### 3.1 Document Retrieve

作者采用的是经典的信息检索方法：

1. term vector model scoring
2. a simple inverted index

具体实现作者在论文中没说，参考代码和比人的总结，其流程大致如下：

1. 对语料单词进行清洗，包括去停词等过滤操作
2. 统计所有的bigram，并对bigram做清洗得到最终的bigram
3. 将这些bigram进行murmur3 hashing得到每个bigram的id（如果哈系特征数目设置过小，可能会有两个不同bigram的id相同，文章用了特征数目为2^24^，可以尽量避免这种哈希冲突）
4. 根据TF-IDF公式计算每个bigram的IDF向量以及TF向量，将IDF乘以TF得到TF-IDF向量
5. 将问题的TF-IDF向量与文章的TF-IDF向量相乘取最大的前五个的文章的索引，得到与问题最相关的5篇文章



### 3.2 Document Reader

作者的Document Reader模块是基于神经网络模型的，轮流对每个段落应用RNN模型，得到预测的答案。

符号说明：

1. 一个问题$q$由$l$个token组成，记$\left\{ q_{1},q_{2}, \cdots, q_{l} \right\} $
2. 文档检索得到的文档有$n$个段落；
3. 一个段落$p$包含了$m$个token，记$\left\{ p_{1},p_{2}, \cdots, p_{m} \right\} $

Document Reader的实现流程为目标段落编码、问题编码、答案预测。

#### 3.2.1 Paragraph encoding

1. 将段落中的每个token $p_{i}$转换成特征向量$\tilde { \mathbf{p}}_{i} \in \mathbb{R}^{d} $；

2. 将特征向量作为输入传入LSTM，得到$p_{i}$的上下文特征向量$\mathbf{p}_{i}$，$\mathbf{p}_{i}$取的是各隐层单元：
   $$
   \left\{ \mathbf{p}_{1}, \cdots, \mathbf{p}_{m} \right\} = \text{RNN}(\left\{\tilde { \mathbf{p}}_{1}, \cdots, \tilde { \mathbf{p}}_{m}\right\})
   $$
   

$\tilde { \mathbf{p}}_{i}$这里作者不仅仅用了传统的Embedding，其组成部分如下，==作者也针对这块做了相应的实验==

- **Word embeddings:**

  $f_{emb}(p_{i})=\mathbf{E}(p_{i})$

  使用训练好的300维Glove词向量，保留绝大多数词向量，对出现频率最高的1000个单词进行fine-tune，比如常见的who, when, how, what, where，这些单词对于QA系统十分关键；

- **Exact match:**

  $f_{exact\_match}(p_{i})=\mathbb{I}(p_{i} \in q)$

  使用3个简单的二值特征（==binary features，什么意思？？==）考虑段落中的token是否匹配问题中单词，不考虑单词得原始形式、大小写

- **Token features：**

  $f_{token}(p_{i})=(\text{POS}(p_{i}),\text{NER}(p_{i}),\text{TF}(p_{i}))$

  考虑了词性(POS)、命名实体识别(NER)、正则的TF

- **Aligned question embedding:**

  $f_{align}(p_{i})=\sum_{j}a_{i,j}\mathbf{E}(q_{j})$

  用来描述段落中的单词和问题中的单词的对齐Embedding，其实就是用attention score $a_{i,j}$ 对问题token的Embedding进行加权求和。

  attention score $a_{i,j}$描述的是$p_{i}$和$q_{j}$之间的相似度，作者的计算方法如下：

  <img src="attention score a_ij.jpg" alt="attention score a_ij" style="zoom:50%;" />

  将每个embedding $\mathbf{E}$经过一层ReLU激活函数的全连接网络，各自相乘并且归一化。

  作者这么做的目的是对*Exact math*的一个补充，比如：car和vehicle两个单词虽然长得完全不一样，但意思却是相似的。这一结论通过实验可以明显得出。



#### 3.2.2 Question encoding

对问题的编码，作者又训练了一个神经网络：
$$
\left\{ \mathbf{q}_{1}, \cdots, \mathbf{q}_{l} \right\} = \text{RNN}(\left\{ \mathbf{E}(q_{1}), \cdots,  \mathbf{E}(q_{l}) \right\})
$$
其中$\mathbf{q}_{i}$是模型的隐层单元，然后对$l$个$\mathbf{q}_{i}$加权求和得到一个向量$\mathbf{q}=\sum_{j}b_{j}{\mathbf{q}_{j}}$，其中$b_{j}$是每个问题单词得重要程度的权重值，计算如下：

<img src="weight b_j.jpg" alt="weight b_j" style="zoom:50%;" />

其中，$\textbf{w}$是模型要学习的权重向量



#### 3.3.3 Prediction

首先，问题答案的预测是在段落级别上的，关键是找到答案在段落中的起始位置。

作者训练了两个分类器，输入是段落向量$\left\{ \mathbf{p}_{1}, \cdots, \mathbf{p}_{m} \right\}$和问题向量$\mathbf{q}$，通过两个带有exp函数的线性网络分别计算每个字符成为开始字符和结束字符的概率：

<img src="Predict answer span.jpg" alt="Predict answer span" style="zoom:45%;" />

在预测过程中，答案的最佳起始位置token$ i$ 和 token$i'$为：
$$
(i,i')=\underset{i,i'}{\text{argmax}}\space P_{start}(i) \times P_{end}(i') \\ s.t. \space i \leq i' \leq i+15
$$
==Q：作者说训练了两个分类器，一个用来判断开始字符，一个是用来判断结束字符的吗？==



## 4. 数据集说明

用了3类数据集：

1）Wikipedia，作为问题答案的知识源；

2）SQuAD Dataset，是作者训练Document Reader的主要数据集；

3）CuratedTREC， WebQuestions，WikiMovies主要用于DrQA的测试和评估多任务学习（Multitask learning）和远程监督（Distant supervision，DS）的效果。



### 4.1 Wikipedia (Knowledge Source)

作为问题答案的知识源；

只保留文字，一共5075182篇文章和9008962个不同的字符



### 4.2 SQuAD

SQuAD数据集主要被用来训练Document Reader。

SQuAD是斯坦福大学于2016年推出基于维基百科的机器阅读数据集。训练集包含87k个样本，开发集包含10k个样本。它的每个样本包含一个自然段和问题和答案。通常用exact string match(EM)与F1 score两种评估方法。

本文作者用SQuAD的训练数据集训练和评估Document Reader的机器阅读理解能力，但因为作者研究的问题是开放域的问答任务，因此测试的时候，作者剔除了自然段，仅仅给出问题以及wikipedia数据库，让模型自己去匹配对应的自然段然后找出答案。



### 4.3 Open-domain QA Evaluation Resources

为了增强模型的泛化能力，作者又在另外3个数据集上做了测试：

- CuratedTREC

- WebQuestions

- WikiMovies

  


### 4.4 Distantly Supervised Data

CuratedTREC、WebQuestions、WikiMovies这三个数据集是问题-答案对的形式，没有关联的文档或段落，因此无法直接用来训练Document Reader.作者借鉴了关系提取中的[distant supervision](#7.1 Distant Supervision)方法：通过关联段落，生成样本，从而丰富训练集。通过Question-Answer对构建训练集的步骤：

1. 基于数据集中的问题，使用document retriever提取相关性最高的5篇文章；
2. 对于五篇文章中的所有段落，抛弃不包含与已知答案完全匹配（no exact match）的段落，抛弃小于25个字大于1500个字的段落，若有的段落中包含命名实体，抛弃那些不包含命名实体的段落；
3. 对于留下来的所有段落，从段落中找出包含答案的span，这里是基于word水平的，也就是unigram，首先找到段落中包含答案的完整区间[start, end]，然后基于20 token window，从start向左延展20个word，从end向右延展20个word（要保证左右两边不能溢出，溢出则取边界）得到一个备选小段落
4. 从备选小段落中找出最有可能的5个小段落，要与问题进行比较。分别对每个小段落统计它的bigram，同时也统计问题的bigram，最后统计每个小段落的bigram与问题的bigram有多少交集，最后对交集求和，求和总数最多的5个小段落为最有可能的段落。





## 5. 实验及结果

作者先分别对Document Retrieve和Document Reader分别做实验，然后基于Wikipedia对DrQA进行系统测试。

### 5.1 测试Document Retrieve

作者在4个数据集上分别用Wiki Search和自己的Document Retrieve检索与问题答案最相关的5篇文章，得到如下结果：

<img src="Document Retrieve Result.jpg" alt="Document Retrieve Result" style="zoom:50%;" />

结果表明:

1. 作者的方法优于Wiki Search；
2. 如果引入bigram，检索效果更好。



### 5.2 测试Document Reader

作者用SQuAD的测试集对Document Reader及其他模型进行了对照试验：

<img src="Evaluation results on the SQuAD dataset.jpg" alt="Evaluation results on the SQuAD dataset" style="zoom:50%;" />

试验结果表明：作者的模型相对更简单，但结果EM和F1都比其他系统要好。

同时，作者在[3.2.1 Paragraph encoding](#3.2.1 Paragraph encoding)中构建token的特征时，分别计算了4种Embedding，它们对模型的贡献程度如下表：

<img src="Feature ablation analysis result.jpg" alt="Feature ablation analysis result" style="zoom:50%;" />

实验结果分析：

> 只去除$f_{align}$这个特征对模型表现影响不大，但是同时去除$f_{align}$和$f_{exact\_match}$模型的表现就会大幅度下降，这可能是因为两者的作用相似又互补。



### 5.3 对DrQA整体实验

作者在前面提到的数据集上对三个版本的DrQA进行了对照试验：

- SQuAD：

  仅用SQuAD训练集训练模型，然后在其他数据集上测试；

- Fine-tune (DS)：

  先用SQuAD预训练好一个模型，然后其他数据集上测试时，进行微调；

- Multitask (DS)：

  基于SQuAD和其它所有DS数据集训练的Document Reader

<img src="Full Wikipedia Result.jpg" alt="Full Wikipedia Result" style="zoom:50%;" />

实验结果表明：

1. 第三个版本的DrQA准确率最高；
2. 版本2和版本3的DrQA相对版本1的QrQA都有明显提升，说明主要的提升原因不是task transfer，而是额外加入 的训练样本。



## 6. 结论

本文基于MSR任务，利用维基百科构建了一个开放域的问答系统DrQA，由document retriever和document reader两个部分组成，分别负责文章提取和阅读理解。实验结果表明，引入多任务学习以及远距离监督（DS）的模型效果最好。



但也存在两个明显值得改进的地方：

1. 对文档的理解需要遍历每个段落，而不能直接对文档进行训练；
2. 文档检索和文档理解还是两块相对独立的模块，不能执行端到端的训练。



## 7. 背景知识介绍

### 7.1 Distant Supervision

**远程监督算法**，是关系抽取任务中目前比较常用的一类方法，应该是*Distant supervision for relation extraction without labeled data*首次提出的概念。

Distant Supervision主要通过将知识库与非结构化文本对齐来自动构建大量训练数据，减少模型对人工标注数据的依赖，增强模型跨领域适应能力。

Distant Supervision的提出主要基于以下假设：

> 两个实体如果在知识库中存在某种关系，则包含该两个实体的非结构化句子均能表示出这种关系。例如，"Steve Jobs", "Apple"在 Freebase 中存在 founder 的关系，则包含这两个实体的非结构文本“Steve Jobs was the co-founder and CEO of Apple and formerly Pixar.”可以作为一个训练正例来训练模型。



这类数据构造方法的具体实现步骤是：

- 从知识库中抽取存在关系的实体对

- 从非结构化文本中抽取含有实体对的句子作为训练样例



Distant Supervision的方法虽然从一定程度上减少了模型对人工标注数据的依赖，但该类方法也存在明显的缺点：

- 假设过于肯定，难免引入大量的**噪声数据**

  如 "Steven Jobs passed away the daybefore Apple unveiled iPhone 4s in late 2011."这句话中并没有表示出 Steven Jobs 与 Apple 之间存在 founder 的关系。

- 数据构造过程依赖于NER等NLP工具，中间过程出错会造成错误传播问题。针对这些问题，目前主要有四类方法：（1）在构造数据集过程中引入先验知识作为限制；（2）利用指称与指称间关系用图模型对数据样例打分，滤除置信度较低的句子；（3）利用多示例学习方法对测试包打标签；（4）采用 attention 机制对不同置信度的句子赋予不同的权值。



### 7.2 Multitask learning

 [模型汇总-14 多任务学习-Multitask Learning概述](https://zhuanlan.zhihu.com/p/27421983)



### 7.3 KBs

知识库根据知识的来源可分为两类：

1. Curated KBs

   - 以Freebase KB和Yago2为代表

   - 他们从维基百科和WordNet等知识库中抽取大量的实体及实体关系，可以把它们理解为是一种**结构化**的维基百科

2. Extracted KBs

   - 以Open Information Extraction（Open IE）和Never-Ending Language Learning（NELL）为代表

   - 这类知识库直接从上亿个网页中抽取实体关系三元组

与Curated KBs相比，Extracted KBs得到的知识更加具有**多样性**，当然，直接从网页中抽取出来的知识，也会存在一定的noisy，其精确度要低于Curated KBs。



## 8. 参考资料

1. [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)
2. [PaperWeekly 第31期 | 远程监督在关系抽取中的应用 ](https://www.sohu.com/a/131451458_500659)
3. [揭开知识库问答KB-QA的面纱1·简介篇](https://zhuanlan.zhihu.com/p/25735572)
4. [论文笔记：Reading Wikipedia to Answer Open-Domain Questions](https://www.zmonster.me/2019/08/07/drqa.html)
5. [【笔记1-2】基于维基百科的开放域问题问答系统DrQA](https://blog.csdn.net/cindy_1102/article/details/88599266)

