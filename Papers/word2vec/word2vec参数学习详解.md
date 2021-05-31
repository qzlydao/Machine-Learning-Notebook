[toc]

## 前言

word2vec虽然非常流行和被广泛关注，但即使在原作者(Mikolov et al)的文章中，也没有给出CBOW和Skip-Gram两个模型的具体推导。同时涉及到的优化算法：hierarchical softmax 和 negative sampling也没有相应的数学推导。

Xin Rong在《word2vec Parameter Learning Explained》给出了了CBOW和Skip-Gram模型以及优化技巧：Hierarchical Softmax和Negative Sampling的详细公式推导，并加以理解。文章由浅入深、循序渐进，是深入理解word2vec的好文章。很痛心的是，Xin Rong于2019年因飞机失事而去世。



## 1. CBOW模型

### 1.1 一个上下文词情况

作者先从CBOW模型的简化版本(即只有一个上下文词，预测目标词)出发，对模型及参数更新进行了推导。

模型图如下：

<img src="/Users/liuqiang/Documents/Machine_Learning/我的文章/word2vec/A simple CBOW model with only one word in the context.jpg" alt="A simple CBOW model with only one word in the context" style="zoom:50%;" />



符号说明：

1. $V$代表词表大小；
2. $N$代表隐藏层的大小；
3. $\mathbf{W}$和$\mathbf{W'}$是两个完全不同的矩阵，不是转置矩阵！！
4. 输入向量$\mathbf{x}$是一个词的ont-hot向量，shape = $V \times 1$

$$
\mathbf{x}=\left[ \begin{matrix} x_{ 1 } \\ x_{ 2 } \\ \vdots  \\ x_{ V } \end{matrix} \right]
$$

5. Input layer到Hidden layer的权重矩阵是一个$V \times N$的矩阵，记为$\mathbf{W}$；

   它的行向量其实就是我们要求的词向量（见下文推导，可以理解为一种降维计算，将原来稀疏的one-hot向量降维为稠密的$N$为向量）

$$
\mathbf{W}=\begin{bmatrix} w_{ 11 } & w_{ 12 } & \cdots  & w_{ 1N } \\ w_{ 21 } & w_{ 22 } & \cdots  & w_{ 2N } \\ \vdots  & \vdots  & \vdots  & \vdots  \\ w_{ V1 } & w_{ V2 } & \cdots  & w_{ VN } \end{bmatrix}=\begin{bmatrix} \mathbf{v}_{ w_{1} } \\ \mathbf{v}_{ w_{2} } \\ \vdots  \\ \mathbf{v}_{ w_{V} } \end{bmatrix} \\
\space \\
\mathbf{W}^{T}=\begin{bmatrix} w_{ 11 } & w_{ 21 } & \cdots  & w_{ V1 } \\ w_{ 12 } & w_{ 22 } & \cdots  & w_{ V2 } \\ \vdots  & \vdots  & \vdots  & \vdots  \\ w_{ 1N } & w_{ 2N } & \cdots  & w_{ VN } \end{bmatrix}=\begin{bmatrix} \textbf{v}_{ w_{1} }^{T} & \textbf{v}_{ w_{2} }^{T} & \cdots & \textbf{v}_{ w_{V} }^{T} \end{bmatrix}
$$



6. Hidden layer到Output layer的权重矩阵为$\mathbf{W'}$， shape = $N \times V$

   $\mathbf{W'}$的列向量$\mathbf{v}_{wj}^{'}$其实也是词向量的一种表达（见下文推导）
   $$
   \mathbf{W'}=\begin{bmatrix} w_{ 11 }^{ ' } & w_{ 12 }^{ ' } & \cdots  & w_{ 1V }^{ ' } \\ w_{ 21 }^{ ' } & w_{ 22 }^{ ' } & \cdots  & w_{ 2V }^{ ' } \\ \vdots  & \vdots  & \vdots  & \vdots  \\ w_{ N1 }^{ ' } & w_{ N2 }^{ ' } & \cdots  & w_{ NV }^{ ' } \end{bmatrix}=\begin{bmatrix} { \mathbf{v} }_{ w_{1} }^{ ' } & { \mathbf{v} }_{ w_{2} }^{ ' } & \cdots  & { \mathbf{v} }_{ w_{V} }^{ ' } \end{bmatrix}
   $$

#### 1.1.1 前向传播

**1. Input layer ——> Hidden layer**

给定一个上下文词的one-hot向量，假设其第$k$个元素$x_{k}=1$，其它元素$x_{k^{'}} =0 \space (k^{'} \neq k)$，则Hidden layer可表示为：
$$
\mathbf{h} = \mathbf{W}^{T}\mathbf{x}=\mathbf{W}_{(k,\cdot)}^{T}=\mathbf{v}_{w_{I}}^{T} \tag{1}
$$ { }
因为$\mathbf{x}$是独热向量，所以Hidden layer $\mathbf{h}$ 可以看做是$\mathbf{W}$的第$k$行的拷贝。



**2. Hidden layer ——> Output layer**

分两步操作：

1. 计算输入词再字典中的score（词向量从$N$到$V$维的线性变换）;
2. 构造条件概率函数（建立词向量与条件概率的关系，有了条件概率，就可以写出最大似然目标函数）



这里先单独考察词表中的任意一个词的得分$u_{j}$（先不直接计算$\mathbf{W'}^{T}\mathbf{h}$）
$$
u_{j}={\mathbf{v}_{w_{j}}^{’}}^{T}\mathbf{h} \space  \space  \space \text{for}  \space  \space j=1,2,\cdots, V \tag{2}
$$
利用softmax函数，可以得到词的后验概率：
$$
p(w_{j}|w_{I})=y_{j}=\frac{\text{exp}(u_{j})}{\sum_{j^{\prime}}^{V}\text{exp}(u_{j^{\prime}})} ,\space \space j=1,2, \cdots , V \tag{3}
$$
$y_{j}$是output layer的第$j$个元素。



将式（1）和（2）带入式（3），得到：
$$
p(w_{j}|w_{I})=\frac{\text{exp}({\mathbf{v}_{w_{j}}^{’}}^{T}\mathbf{v}_{w_{I}})}{\sum_{j^{\prime}=1}^{V}\text{exp}({\mathbf{v}_{w_{j^{\prime}}}^{’}}^{T}\mathbf{v}_{w_{I}})} 
,\space \space j=1,2, \cdots , V
\tag{4}
$$
理解：

> 1. $\mathbf{v}_{w}$和$\mathbf{v}_{w}^{\prime}$都是词$w$的向量表示；
> 2. $\mathbf{v}_{w}$是权重矩阵$\mathbf{W}$的行向量；$\mathbf{v}_{ w }^{\prime}$是权重矩阵$\mathbf{W}^{\prime}$的行向量；
> 3. $\mathbf{v}_{w}$称之为**input vector**， $\mathbf{v}_{w}^{\prime}$称之为**output vector**；
> 4. ==$\mathbf{v}_{w_{j}}^{\prime}$和$\mathbf{v}_{w_{I}}$越相似，则条件概率越大==



#### 1.1.2 反向传播更新参数

写出优化目标函数，利用反向传播算法更新参数（由式(4)可以看出，目标函数的参数应该是$\mathbf{v}_{w_{I}}$和$\mathbf{v}_{w_{j}}^{\prime}$）



==设真实的目标值是$w_{O}$，它在output layer的索引是$j^{*}$==，则目标函数可以表示为：
$$
\begin{split}
\text{maximize} \space p(w_{O}|w_{I}) &=\text{max} \space y_{j^{*}} \\
& = \text{max log}\space y_{j^{*}} \\
& = u_{j^{*}}-\text{log} \space \sum_{j^{\prime}=1}^{V} \text{exp}(u_{j^{\prime}}):=-E
\end{split} \tag{7}
$$
这里，$E=\text{log}\space p(w_{O}|w_{I})$，这个损失函数可以理解为一种特殊的[交叉熵](https://www.jianshu.com/p/7171703fade1)。

**1. 有了目标函数，首先推导$\mathbf{v}_{w_{j}}^{\prime}$的更新更新公式**
$$
\frac{\partial E}{\partial u_{j}}=y_{j}-t_{j}:=e_{j}, \space \space j=1,2,\cdots, V \tag{8}
$$
这里，$t_{j}=\mathbb{I}(j=j^{*})$，只有当第$j$个unit代表的是真实的输出词时，$t_{j}$才等于1，否则$t_{j}=0$

接着，利用链式求导法则，可以求出$E$关于矩阵$\mathbf{W^{\prime}}$中元素${w_{i,j}^{\prime}}$ 的偏导数：
$$
\frac {\partial E}{\partial w_{ij}^{\prime}}=\frac {\partial E}{\partial u_{j}}\cdot \frac{\partial u_{j}}{\partial w_{ij}^{\prime}}=e_{j} \cdot h_{i} \tag{9}
$$
利用随机梯度下降法，可以得到Hidden layer到output layer的权重更新公式如下：
$$
\begin{eqnarray*}
{w_{ij}^{\prime}}^{(new)}={w_{ij}^{\prime}}^{(old)}-\eta \cdot e_{j} \cdot h_{i} \tag{10}\\
or \\
{\textbf{v}_{w_{j}}^{\prime}}^{(new)}={\textbf{v}_{w_{j}}^{\prime}}^{(old)}-\eta \cdot e_{j} \cdot \textbf{h} \space\space\space\space\space \text{for} \space j = 1,2,\cdots,V \tag{11}
\end{eqnarray*}
$$
更新公式理解：

> 1. $\eta > 0$，学习率；$e_{j}=y_{j}-t_{j}$，$\textbf{h}=\textbf{v}_{w_{I}}$
> 2. 通过更新公式可以看出，==每一次更新需遍历词表中的每一个单词（弊端）==，计算出它的输出概率$y_{j}$，并和期望输出$t_{j}（0 \space \text{or} \space 1）$进行比较：
>    - 如果$y_{j} > t_{j} \Rightarrow \space e_{j}>0$，("overestimating"，这种情况是：$t_{j}=0$, 即输出的第$j$个单词并不是真实的上下文词)，那么就从$\textbf{v}_{w_{j}}^{\prime}$中减去隐藏向量$\textbf{h}$中的一部分（比如$\textbf{v}_{w_{I}}$），这样向量$\textbf{v}_{w_{j}}^{\prime}$就会与向量$\textbf{v}_{w_{I}}$相差更远；
>    - 如果$y_{j} < t_{j} \Rightarrow \space e_{j}<0$，("underestimating"，这种情况是：$t_{j}=1$，即输出的第$j$ 个词确实是真实的上下文词)，那么就从$\textbf{v}_{w_{j}}^{\prime}$中加上隐藏向量$\textbf{h}$中的一部分，这样向量$\textbf{v}_{w_{j}}^{\prime}$就会与向量$\textbf{v}_{w_{I}}$更接近；
>    - 如果$y_{j}$和$t_{j}$非常接近，那么更新参数基本上没啥变化。

这里再强调一遍，$\textbf{v}_{w}^{\prime}$和$\textbf{v}_{w}$是单词$w$的两种不同的向量表示形式。



**2. 介绍完输出向量$\textbf{v}_{w}^{\prime}$的更新公式后，接下来介绍输入向量$\textbf{v}_{w}$的更新公式**

首先，对隐藏层单元$h_{i}$求偏导：
$$
\frac{\partial E}{\partial h_{i}}=\sum_{j=1}^{V}\frac{\partial E}{\partial u_{j}} \cdot \frac{\partial u_{j}}{\partial h_{i}}=\sum_{j=1}^{V}e_{j} \cdot w_{ij}^{\prime}:=\text{EH}_{i} \space \space \space \space \text{for} \space i=1,2,\cdots, N \tag{12}
$$
$\text{EH}$是一个$N$维向量。

将式（1）$\mathbf{h} = \mathbf{W}^{T}\mathbf{x}=\mathbf{W}_{(k,\cdot)}=\mathbf{v}_{w_{ I }}^{T}$展开：
$$
h_{i} = \sum_{k=1}^{V}x_{k} \cdot w_{ki} \tag{13}
$$
于是，利用链式求导法则，可得到$E$关于$w_{ki}$的偏导数：
$$
\frac{\partial E}{\partial w_{ki}}=\frac{\partial E}{\partial h_{i}} \cdot \frac{\partial h_{i}}{\partial w_{ki}}=\text{EH}_{i} \cdot x_{k} \tag{14}
$$
利用张量乘积的方式，可以得到：
$$
\frac{\partial E}{\partial \textbf{W}}=\begin{bmatrix} \text{EH}_{1}\cdot x_{1} & \text{EH}_{2}\cdot x_{1} & \cdots & \text{EH}_{N}\cdot x_{1} \\ \text{EH}_{1}\cdot x_{2} & \text{EH}_{2}\cdot x_{2} & \cdots & \text{EH}_{N}\cdot x_{2} \\ \vdots & \vdots & \vdots & \vdots \\ \text{EH}_{1}\cdot x_{V} & \text{EH}_{2}\cdot x_{V} & \cdots & \text{EH}_{2}\cdot x_{V} \end{bmatrix}=\mathbf{x}\otimes \text{EH}=\mathbf{x}\text{EH}^{T} \tag{15}
$$
对上式分析可得：

> 1. 得到一个$V \times N$矩阵；
> 2. 因为$\mathbf{x}$中只有一项不等于0，所以矩阵$\frac{\partial E}{\partial \mathbf{W}}$中只有一行不等0，其它行都为0；


于是，得到$\mathbf{W}$的更新公式为：
$$
\textbf{v}_{w_{I}}^{(new)}=\textbf{v}_{w_{I}}^{(old)}-\eta \cdot \text{EH}^{T} \tag{16}
$$
这里，$\textbf{v}_{w_{I}} $是矩阵$\textbf{W}$的行向量，是唯一的上下文词的"input vector"，也是矩阵中唯一的导数不等于0的行。其它行元素不会更新，因为它们的导数为0。



更新公式理解：

> $\text{EH}_{i}=\sum_{j=1}^{V}e_{j} \cdot w_{ij}^{\prime}$，可以看到$\text{EH}$向量其实是字典中每个词的output vector的加权求和（权重为预测误差$e_{j}=y_{j}-t_{j}$），所以：
>
> 1. 如果输出的第 $j$ 个词$w_{j}$ 被高估了（即，$y_{j}>t_{j}$），那么输入词向量$w_{I}$就会远离这个输出词向量$w_{j}$(即 $w_{I} \cdot w_{j} >>0$)；
> 2. 相反地，如果输出的词$w_{j}$被低估了，(即，$y_{j}<t_{j}$)，那么输入词向量$w_{I}$就会靠近$w_{j}$($w_{I} \cdot w_{j} >>1$)；
> 3. 如果$w_{j}$的概率很准确，那么$w_{I}$几乎不怎么变。



### 1.2 多个上下文词的CBOW模型

多个上下文词的CBOW模型如下图所示：

<img src="Continuous bag-of-word model.jpg" alt="Continuous bag-of-word model" style="zoom:50%;" />

因为有多个上下文词，隐藏层的输出向量$\mathbf{h}$取得是每个词向量乘权重矩阵$\textbf{W}$后的平均：
$$
\begin{eqnarray*}
\textbf{h} &=& \frac{1}{C}\textbf{W}^{T}(\textbf{x}_{1}+\textbf{x}_{2}+\cdots+\textbf{x}_{C}) \tag{17} \\
&=& \frac{1}{C}(\textbf{v}_{w_{1}}+\textbf{v}_{w_{2}}+\cdots+\textbf{v}_{w_{C}}) \tag{18}
\end{eqnarray*}
$$
符号说明：

> 1. $w_{1},w_{2},\cdots,w_{C}$为$C$个上下文词；
> 2. $\textbf{x}_{w}$是词$w$的输入向量



损失函数：
$$
\begin{split}
E &= -p(w_{O}|w_{I,1},w_{I,2},\cdots, w_{I,C}) \\
&= -u_{j^{*}} + \text{log} \space \sum_{j^{\prime}=1}^{V} \text{exp}(u_{j^{\prime}}) \\
&= -{\mathbf{v}_{w_{O}}^{\prime}}^{T} \cdot \color{red}{\mathbf{h}} + \text{log} \sum_{j^{\prime}=1}^{V} \text{exp}({\mathbf{v}_{w_{j}}^{\prime}}^{T} \cdot \color{red}{\mathbf{h}})
\end{split} \tag{21}
$$
目标函数与1.1中一个词的情况是一样的，除了$\mathbf{h}$的定义不一样。



权重矩阵$\mathbf{W'}$的更新公式也不变：
$$
{\textbf{v}_{w_{j}}^{\prime}}^{(new)}={\textbf{v}_{w_{j}}^{\prime}}^{(old)}-\eta \cdot e_{j} \cdot \textbf{h} \space\space\space\space\space \text{for} \space j = 1,2,\cdots,V \tag{22}
$$

<span name="23"> 权重矩阵 </span>$\mathbf{W}$的更新公式也类似：
$$
\textbf{v}_{w_{I,c}}^{(new)}=\textbf{v}_{w_{I,c}}^{(old)}-\frac{1}{C} \cdot \eta \cdot \text{EH}^{T} \space \space \space \text{for} \space \space c=1,2,\cdots,C \tag{23}
$$



## 2. Skip-Gram模型

Skip-Gram模型正好与CBOW相反，它是根据中心词预测上下文词，模型示意图如下：

<img src="The skip-gram model.jpg" alt="The skip-gram model" style="zoom:50%;" />

#### 2.1.1 前向传播

符号定义还是不变，
$$
\mathbf{h} = \mathbf{W}^{T}\mathbf{x}=\mathbf{W}_{(k,\cdot)}^{T}=\mathbf{v}_{w_{I}}^{T} \tag{24}
$$
在输出层，skip-gram模型的输出是$C$个多项式分布，但它们共用一个权重矩阵$\textbf{W}_{N \times V}^{\prime}$
$$
p(w_{c,j}=w_{O,c}|w_{I})=y_{c,j}=\frac{\text{exp}(u_{c,j})}{\sum_{j^{\prime}=1}^{V} \text{exp}(u_{j^{\prime}})} \tag{25}
$$
符号说明：

> 1. $w_{c,j}$是输出层第$c$个pannel(有几个上下文词，就有几个pannel)的第$j$ 个词；
> 2. $w_{O,c}$是输出上下文词中的第$c$ 个词；
> 3. $w_{I}$是唯一的输入单词；
> 4. $y_{c,j}$为输出层的第$c$个panel上的第$j$个神经单元的概率输出值；
> 5. $u_{c,j}$是输出层第$c$ 个pannel上的第$j$ 个神经元的输入值（score）



注意到输出层所有pannel都共享一个权重矩阵$\mathbf{W'}$，因此每个pannel的输入$u_{c,j}$都是一样的：
$$
u_{c,j}=u_{j}={\textbf{v}_{w_{j}}^{\prime}}^{T} \cdot \textbf{h}, \space \space \text{for} \space c=1,2,\cdots, C \tag{26}
$$


#### 2.1.2 反向更新参数

损失函数定义为：
$$
\begin{eqnarray*}
E &=& -\text{log}\space p(w_{O,1},w_{O,2},\cdots,w_{O,C}|w_{I})  \\
 &=& -\text{log}\space \prod_{c=1}^{C}{\frac{\text{exp}(u_{c,j_{c}^{*}})}{\sum_{j^{\prime}}^{V}\text{exp}(u_{j^{\prime}})}} \\
 &=&-\sum_{c=1}^{C}{u_{j_{c}^{*}}}+C\cdot \text{log}\sum_{j'}^{V}\text{exp}(u_{j'}) \tag{29}
\end{eqnarray*}
$$

> 说明：$j_{c}^{*}$是输出层第$c$个pannel(也即是第$j$个上下文词)在字典中的索引。





对$E$求$u_{c,j}$的偏导：
$$
\frac{\partial E}{\partial u_{c,j}}=y_{c,j}-t_{c,j}:=e_{c,j} \tag{30}
$$
为了表示方便，作者定义了一个$V$维向量$\text{EI}=\left\{ \text{EI}_{1},\text{EI}_{2},\cdots,\text{EI}_{V} \right\} $表示每一个上下文词的预测误差之和：
$$
\text{EI}_{j}=\sum_{c=1}^{C}e_{c,j} \tag{31}
$$


接下来，求损失函数$E$关于$\text{W}^{\prime} $的偏导数：
$$
\begin{eqnarray*}
\frac{\partial E}{\partial w_{i,j}^{\prime}} &=& \sum_{c=1}^{C}\frac{\partial E}{\partial u_{c,j}} \cdot \frac{\partial u_{c,j}}{\partial w_{i,j}^{\prime}} = \sum_{c=1}^{C}e_{c,j}\cdot h_{i} \\
&=& h_{i} \cdot \sum_{c=1}^{C}e_{c,j} =\text{EI}_{j} \cdot h_{i}
\end{eqnarray*} \tag{32}
$$
==从而，可以得到$\text{W}^{\prime}$（$N \times V$维）的更新公式为：==
$$
{w_{ij}^{\prime}}^{(new)}={w_{ij}^{\prime}}^{(old)}-\eta \cdot \text{EI}_{j} \cdot h_{i} \tag{33}
$$
or
$$
{\text{v}_{w_{j}}^{\prime}}^{(new)}={\text{v}_{w_{j}}^{\prime}}^{(old)}-\eta \cdot \text{EI}_{j} \cdot \text{h} \space \space \space \text{for} \space j=1,2,\cdots,V \tag{34}
$$

>  说明：
>
>  1. 对更新公式的理解和式(11)一样，只是输出层的预测误差是基于$C$个上下文词；
>  2. 对每一个训练样本，需要利用该更新公式更新$\text{W}^{\prime}$的每一项。



Input layer到Hidden layer的<span name="35">权重矩阵</span>$\text{W}$的更新公式与(16)一样：
$$
\begin{eqnarray*}
\textbf{v}_{w_{I}}^{(new)}=\textbf{v}_{w_{I}}^{(old)}-\eta \cdot \text{EH}^{T} \tag{35} \\
\text{where} \space \space  \space  \space  \space  \space  \space  \text{EH}_{i}=\sum_{j=1}^{V}\text{EI}_{j} \cdot w_{ij}^{\prime} \tag{36}
\end{eqnarray*}
$$




## 3. 计算优化

### 3.1 问题分析

前面两节讨论了CBOW和Skip-Gram两个模型的原始形式。

为了更好的引出优化技巧，对模型示意图中的output layer详细画了下：

<img src="modify cbow model.png" alt="modify cbow model" style="zoom:50%;" />



对模型的原始形式进行分析：

1. 模型的参数是input vector $\mathbf{v}_{w}$和output vector $\mathbf{v}_{w}^{\prime }$，它们其实是一个词的两个表示向量；
2. 参数$\mathbf{v}_{w} $的学习成本不高，但是$\mathbf{v}_{w}^{\prime} $的学习成本(时间复杂度)就很高了，分析如下：



首先回顾下更新$\mathbf{v}_{w}^{\prime} $时涉及到的计算：
$$
\begin{cases} {\text{v}_{w_{j}}^{\prime}}^{(new)}={\text{v}_{w_{j}}^{\prime}}^{(old)}-\eta \cdot e_{j} \cdot \textbf{h} \\ e_{j}=y_{j}-t_{j} \\ y_{j}=\frac{\text{exp}(u_{j})}{\color{red}{\sum_{j^{\prime}}^{V}\text{exp}(u_{j^{\prime}})}} \\ u_{j}={\mathbf{v}_{w_{j}}^{’}}^{T}\mathbf{h} \end{cases} \space \space \space \space \space \space \space \text{for} \space j=1,2,\cdots,V
$$
==通过观察上面的更新计算公式可以发现，对于每一个训练样本，都需要遍历字典中的每一个词$w_{j}$，计算$u_{j}$、$y_{j}$、$e_{j}$，最终更新$\textbf{v}_{w_{j}}^{\prime }$，这个计算开销是很大的！==

为了解决这个问题，一种直观的方法是每次迭代限制需要更新的输出向量，一种有效的实现是使用hierarchical softmax；另外一种方法是通过采样解决。

### 3.2 Hierarchical Softmax

Hierarchical Softmax是计算softmax的一种高效的方法。Hierarchical Softmax将词典构建成一个棵Huffman Tee。

<img src="/Users/liuqiang/Documents/Machine_Learning/我的文章/word2vec/An example binary tree for the hierarchical softmax model.png" alt="An example binary tree for the hierarchical softmax model" style="zoom:50%;" />

关于这棵树模型的说明：

1. $V$个单词必须是在叶节点上；如此，就有$V-1$个内部节点；
2. 对于每个叶节点，只存在一条路径从根节点通向该节点；
3. 到达单词$w$的路径长度记为$L(w)$，如图，$L(w_{2})=4$；
4. $n(w,j)$表示$w$路径上的第$j$个节点；



在Hierarchical Softmax模型中，不再有词的输出向量（output vector）这种表达。而是$V-1$个内部节点都有一个输出向量$\mathbf{v}_{ n(w,j) }^{\prime }$  。因此一个单词作为输出单词的概率计算公式定义如下：：
$$
p(w=w_{O})=\prod_{j=1}^{L(w)-1}{\sigma \left \{ \mathbb{I} \left [ n(w,j+1)=\text{ch}(n(w,j))\right ] \cdot {\textbf{v}_{n(w,j)}^{\prime }}^{T} \cdot \textbf{h} \right \} }  \tag{37}
$$
公式符号说明：

1. $\text{ch}(n)$表示节点$n$的左子节点；
2. $\textbf{v}_{n(w,j)}^{\prime}$是内部节点$n(w,j)$的向量表达；
3. $\textbf{h}$是隐藏层的输出向量（Skip-Gram模型对应$\textbf{h}=\textbf{v}_{w_{I}}$，CBOW模型对应$\textbf{h}=\frac{1}{C}\sum_{c=1}^{C}\textbf{v}_{w_{v}}$）；
4. $\mathbb{I}[x]$指示函数，定义如下：

$$
\mathbb{I}[x]=\begin{cases} 1 \space \space \space \space \space \space \space \space \space \text{if x is true}\\ -1 \space \space \space \space \space \space \text{otherwise }\end{cases} \tag{38}
$$



结合Figure 4对公式（37）的理解：

> 如果定义$w_{2}$是输出词的概率呢？这里将其概率定义为从根节点随机游走到$w_{2}$的概率；
>
> 在每个内部节点（包括根节点），需要定义接下来走到左子树和右子树的概率。这里将在节点$n$处，走向左子树的概率定义为：
> $$
> p(n,left)=\sigma({\textbf{v}_{n}^{\prime}}^{T}\cdot \textbf{h}) \tag{39}
> $$
> 相应地，走向右子树的概率定义为：
> $$
> p(n,right)=1-\sigma({\textbf{v}_{n}^{\prime}}^{T}\cdot \textbf{h})=\sigma(-{\textbf{v}_{n}^{\prime}}^{T}\cdot \textbf{h}) \tag{40}
> $$
> 如此，那么$w_{2}$是输出词的概率为：
> $$
> \begin{eqnarray*}
> p(w_{2}=w_{O}) &=& p(n(w_{2},1),left)\cdot p(n(w_{2},2),left) \cdot p(n(w_{2},3),right) \\
> &=& \sigma({\textbf{v}_{n(w_{2},1)}^{\prime}}^{T}\cdot \textbf{h}) \cdot \sigma({\textbf{v}_{n(w_{2},2)}^{\prime}}^{T}\cdot \textbf{h}) \cdot \sigma(-{\textbf{v}_{n(w_{2},3)}^{\prime}}^{T}\cdot \textbf{h}) \tag{42}
> \end{eqnarray*}
> $$
> 不难证明，所有概率求和等于1：
> $$
> \sum_{i=1}^{V}p(w_{i}=w_{O})=1 \tag{43}
> $$



接下来推导内部节点上的向量的更新公式，首先考虑只有一个上下文词的情况，记符号：
$$
\mathbb{I}[\cdot]:=\mathbb{I}[n(w,j+1)=\text{ch}(n(w,j))] \tag{44}
$$

$$
\textbf{v}_{j}^{\prime}:=\textbf{v}_{n_{w,j}}^{\prime} \tag{45}
$$



给定一个训练样本，误差函数定义为：
$$
E=-\text{log}\space p(w=w_{O}|w_{I})=-\sum_{j=1}^{L(w)-1}\text{log}\space \sigma(\mathbb{I[\cdot]{\textbf{v}_{j}^{\prime}}}^{T}\textbf{h}) \tag{46}
$$
求$E$对${\textbf{v}_{j}^{\prime}}^{T}\textbf{h}$的偏导：
$$
\begin{eqnarray*}
\frac{\partial E}{\partial {\textbf{v}_{j}^{\prime}}^{T}\textbf{h}} &=& \left( \sigma(\mathbb{I[\cdot]{\textbf{v}_{j}^{\prime}}}^{T}\textbf{h})-1 \right)\mathbb{I}[\cdot] \\
&=& \begin{cases} \sigma({\textbf{v}_{j}^{\prime}}^{T}\textbf{h})-1 \space \space\space\space\space (\mathbb{I}[\cdot]=1)\\ \sigma({\textbf{v}_{j}^{\prime}}^{T}\textbf{h}) \space \space\space\space\space\space\space\space\space\space\space\space (\mathbb{I}[\cdot]=-1)\end{cases} \\
&=& \sigma({\textbf{v}_{j}^{\prime}}^{T}\textbf{h})-t_{j} \tag{49}
\end{eqnarray*}
$$
上式中，当$\mathbb{I}[\cdot]=1$时，$t_{j}=1$，否则$t_{j}=0$.

> 补充：$\sigma(x)$函数性质：
>
> 1. $\sigma(x)=\frac{1}{1+e^{-x}}$
>
> 2. $\sigma^{\prime}(x)=\sigma(x)(1-\sigma(x))$
>
> 3. $\sigma(-x)=1-\sigma(x)$
> 4. $\left[ \text{log}\sigma(x) \right ]^{\prime}=1-\sigma(x)$
> 5. $[\text{log}(1-\sigma(x)) ]^{\prime}=-\sigma(x)$



进一步，可以求得损失函数$E$对于内部节点的向量表达$\mathbf{v}_{j}^{\prime}$的偏导：
$$
\frac{\partial E}{\partial \textbf{v}_{j}^{\prime }}=\frac{\partial E}{\partial {\textbf{v}_{j}^{\prime}}\textbf{h}} \cdot \frac{\partial {\textbf{v}_{j}^{\prime}}\textbf{h}}{\partial \textbf{v}_{j}^{\prime }}=\left( \sigma({\textbf{v}_{j}^{\prime}}^{T}\textbf{h})-t_{j} \right)\cdot \textbf{h} \space\space\space\space  \text{for} \space  \space  j=1,2,\cdots,L(w)-1 \tag{50}
$$
因此，可以得到输出向量$\mathbf{v}_{j}^{\prime}$的更新公式：
$$
{\mathbf{v}_{j}^{\prime}}^{(new)}={\mathbf{v}_{j}^{\prime}}^{(old)}-\eta\left( \sigma({\textbf{v}_{j}^{\prime}}^{T}\textbf{h})-t_{j} \right)\cdot \textbf{h} \space\space\space\space  \text{for} \space  \space  j=1,2,\cdots,L(w)-1  \tag{51}
$$
对更新公式的理解：

> 1. $\sigma({\textbf{v}_{j}^{\prime}}^{T}\textbf{h})-t_{j}$可理解为在内部节点$n(w,j)$上的预测误差；
> 2. 学习的“任务”可理解为在随机游走过程中，在某个内部节点上，下一步分别走向左子树和右子树的概率；
> 3. $t_{j}=1$意味着节点的路径指向左子树，$t_{j}=0$意味着节点的路径指向右子树，$\sigma({\textbf{v}_{j}^{\prime}}^{T}\textbf{h})$是预测结果；
> 4. 在训练过程中，$\mathbf{v}_{j}^{\prime}$会根据预测的结果做更新（靠近或远离$\textbf{h} $）；
> 5. 该更新公式同时适用于CBOW模型和Skip-Gram模型，对于Sikp-Gram模型，因为有$C$个上下文词，因此在一次训练过程中，需执行$C$遍更新操作。



推导出$\mathbf{v}_{j}^{\prime}$后，接下来推导输入向量$\mathbf{v}_{w}$的更新公式：
$$
\begin{eqnarray*}
\frac{\partial E}{\partial \textbf{h}} &=& \sum_{j=1}^{L(w)-1} \frac{\partial E}{\partial {\textbf{v}_{j}^{\prime}}\textbf{h}} \cdot \frac{\partial {\textbf{v}_{j}^{\prime}}\textbf{h}}{\partial \textbf{h}} \\
&=& \sum_{j=1}^{L(w)-1} \left( \sigma({\textbf{v}_{j}^{\prime}}^{T}\textbf{h})-t_{j} \right)\cdot \textbf{v}_{j} \\
&:=& \text{EH} \tag{54}
\end{eqnarray*}
$$


带入[式(23)](#23)可以得到CBOW模型输入向量的更新公式：
$$
\textbf{v}_{w_{I,c}}^{(new)}=\textbf{v}_{w_{I,c}}^{(old)}-\frac{1}{C} \cdot \eta \cdot \text{EH}^{T} \space \space \space \text{for} \space \space c=1,2,\cdots,C
$$
带入[式(35)](#35)可以得到Skip-Gram模型的输入向量的更新公式：
$$
\textbf{v}_{w_{I}}^{(new)}=\textbf{v}_{w_{I}}^{(old)}-\eta \cdot \text{EH}^{T}
$$
**总结：**

对比下原始形式和Hierarchical Softmax形式下的$\textbf{v}_{w_{j}}^{\prime}$和$\textbf{v}_{j}^{\prime}$的更新公式，参考公式(2)~(11)和公式(51):

1. 原始形式：

$$
{\textbf{v}_{w_{j}}^{\prime}}^{(new)}={\textbf{v}_{w_{j}}^{\prime}}^{(old)}-\eta \cdot \left( \frac{\text{exp}({\mathbf{v}_{w_{j}}^{’}}^{T}\mathbf{h})}{\sum_{j^{\prime}}^{V}\text{exp}({\mathbf{v}_{w_{j'}}^{’}}^{T}\mathbf{h})}-t_{j} \right)  \cdot \textbf{h}
$$

2. Hierarchical Softmax形式：

$$
{\mathbf{v}_{j}^{\prime}}^{(new)}={\mathbf{v}_{j}^{\prime}}^{(old)}-\eta\left( \sigma({\textbf{v}_{j}^{\prime}}^{T}\textbf{h})-t_{j} \right)\cdot \textbf{h}
$$

可以看到Hierarchical Softmax形式，更新$\textbf{v}_{j}^{\prime } $时，与其它内部节点无关。两个模型在参数几乎相等(原始形式有$V$个单词，Hierarchical Softmax形式有$V-1$个内部节点)的情况下，时间复杂度从$O(V)$降到了$O(\text{log}(V)$.



### 3.3 Negative Sampling

Negative Sampling的思想比较直接：每次迭代的时候，根据采样结果，部分更新输出矩阵。



很显然，正样本应该在我们的样本集合中，同时需要采样一部分词作为负样本。在采样的过程中，我们可以任意选择一种概率分布。我们将这种概率分布称为“噪声分布”（the noise distribution），用$P_{n}(w)$来表示。我们可以根据经验选择一种较好的分布。



在 word2vec中，==作者没有使用一种能够产生良好定义的后验多项式分布的负采样形式(不是特别理解？)，而是使用下面的训练目标函数：==
$$
E=-\text{log}\space \sigma\left( {\mathbf{v}_{w_{O}}^{’}}^{T}\mathbf{h} \right)-\sum_{w_{j}\in \mathcal{W}_{neg}}\text{log}\space \sigma \left( -{\mathbf{v}_{w_{j}}^{’}}^{T}\mathbf{h} \right)  \tag{55}
$$
上式中：

> 1. $w_{O}$是输出词(即正样本)，$\textbf{v}_{w_{O}}^{\prime} $是其对应的输出向量；
> 2. $\textbf{h}$是隐藏层的输出向量：
>    - COBW：        $\textbf{h}=\frac{1}{C} \sum_{c=1}^{C} \textbf{v}_{w_{c}}$；
>    - Skip-Gram：$\textbf{h}=\textbf{v}_{w_{ I }}$
> 3. $\mathcal{W}_{neg}=\left\{ w_{j} | j=1,2,\cdots, K \right\} $负样本集合



接下来的推导与之前并无二致。
$$
\begin{eqnarray*}
\frac{\partial E}{\partial {\textbf{v}_{w_{j}}^{\prime}}^{T}\textbf{h}} &=& \begin{cases} \sigma({\textbf{v}_{w_{j}}^{\prime}}^{T}\textbf{h})-1 \space \space\space\space\space \text{if}\space w_{j}=w_{O}\\ \sigma({\textbf{v}_{w_{j}}^{\prime}}^{T}\textbf{h}) \space \space\space\space\space\space\space\space\space\space\space\space \text{if}\space w_{j} \in \mathcal{W}_{neg} \end{cases} \tag{56} \\ 

&=& \sigma({\textbf{v}_{w_{j}}^{\prime}}^{T}\textbf{h})-t_{j} \tag{57}
\end{eqnarray*}
$$
上式中，当$w_{j}$是正样本时，$t=1$；否则，$t=0$.



进一步对$\mathbf{v}_{w_{j}}^{\prime } $求导：
$$
\frac{\partial E}{\partial {\textbf{v}_{w_{j}}^{\prime}}}
=
\frac{\partial E}{\partial {\textbf{v}_{w_{j}}^{\prime}}^{T}\textbf{h}} 
\cdot 
\frac{\partial {\textbf{v}_{w_{j}}^{\prime}}^{T}\textbf{h}}{\partial {\textbf{v}_{w_{j}}^{\prime}}}
= \left( \sigma({\textbf{v}_{w_{j}}^{\prime}}^{T}\textbf{h})-t_{j} \right)\textbf{h} \tag{58}
$$
因此$\textbf{v}_{w_{j}}^{\prime }$的更新公式为：
$$
{\mathbf{v}_{w_{j}}^{\prime}}^{(new)}={\mathbf{v}_{w_{j}}^{\prime}}^{(old)}-\eta\left( \sigma({\textbf{v}_{w_{j}}^{\prime}}^{T}\textbf{h})-t_{j} \right)\cdot \textbf{h} 
\space \space \space \space \space 
\color{red}{\text{for} \space w_{j} \in \left\{ w_{O} \right\} \cup \mathcal{W}_{neg} }
\tag{59}
$$
每次迭代，只更新样本集中词的输出向量（output vector），而不是更新整个字典中词的输出向量（output vector），因此，能显著提高计算效率。上述更新公式对CBOW和Skip-Gram模型都适用。



继续计算$E$关于$\mathbf{h}$的偏导：
$$
\begin{eqnarray*}
\frac{\partial E}{\partial \textbf{h}} &=& \sum_{\space w_{j} \in \left\{ w_{O} \right\} \cup \mathcal{W}_{neg}} \frac{\partial E}{\partial {\textbf{v}_{w_{j}}^{\prime}}^{T}\textbf{h}} 
\cdot 
\frac{\partial {\textbf{v}_{w_{j}}^{\prime}}^{T}\textbf{h}}{\partial {\textbf{h}}} \\
&=& \sum_{\space w_{j} \in \left\{ w_{O} \right\} \cup \mathcal{W}_{neg}} \left( \sigma({\textbf{v}_{w_{j}}^{\prime}}^{T}\textbf{h})-t_{j} \right)\cdot \textbf{v}_{w_{j}}^{\prime } \\
&:=& \text{EH} \tag{61}
\end{eqnarray*}
$$
将$\text{EH}$带入式(23)得到CBOW模型输入向量(input vector)的更新公式；

将$\text{EH}$带入式(35)得到Skip-Gram模型输入向量(input vector)的更新公式。



## 4. 参考

1. [word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738)

2. [《word2vec Parameter Learning Explained》论文学习笔记](https://blog.csdn.net/lanyu_01/article/details/80097350)













