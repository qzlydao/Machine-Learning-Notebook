[TOC]



## 1. EM介绍

EM(Expectation Maximization Algorithm, EM)是Dempster等人于1977年提出的一种<font color='#c63c26'>迭代算法</font>，用于**含有隐变量的概率模型参数的极大似然估计(MLE)，或极大后验概率估计(MAP)。**



## 2. EM算法描述

1. 输入

   >$X$：观测变量数据
   >
   >$Z$：隐变量数据
   >
   >$P(X,Z|\theta)$：联合分布
   >
   >$P(Z|X,\theta)$：条件分布，后验概率

2. 输出

   > $\hat{\theta}$：模型参数

3. 迭代过程

   - 初始化参数$\theta^{(0)}$

   - $E$步：记$\theta^{(i)}$是第$i$ 次迭代参数$\theta$的估计值，则第$i+1$ 次迭代的$E$步：<font color=red>求对数联合概率在后验上的期望：</font>
     $$
     \begin{eqnarray*}
     Q(\theta,\theta^{(i)}) &=& E_{Z}\left[\text{log}P(X,Z|\theta)|X,\theta^{(i)} \right] \\
     &=& \sum_{Z}\text{log}P(X,Z|\theta)P(Z|X,\theta^{(i)})
     \end{eqnarray*}
     $$

   - $M$步：求$i+1$步的参数估计值$\theta^{(i+1)}$：
     $$
     \theta^{(i+1)}=\underset{\theta}{\text{argmax}}Q(\theta,\theta^{(i)})
     $$

   - 重复$E$步和$M$步，直到收敛：
     $$
     \left\|\theta^{(i+1)} - \theta^{(i)} \right\| < \varepsilon_{1} \\
     \text{or} \\
     
     \left\|Q(\theta^{(i+1)} , \theta^{(i)} ) - Q(\theta^{(i)} , \theta^{(i)} )\right\| < \varepsilon_{2}
     $$
     

## 3. EM公式导出之ELBO+KL Divergence

MLE的做法是最大化似然函数：
$$
\begin{eqnarray*}
\mathcal{L}{(\theta)} &=& \text{log}P(X|\theta)=\text{log}\sum_{Z}P(X,Z|\theta) \\
&=& \text{log} \left\{\sum_{Z}P(X|Z,\theta)P(Z|\theta)  \right\}
\end{eqnarray*}
$$
上面的式子中有隐变量$Z$并且是$\text{log}\sum$形式，不好直接计算。



EM的做法是求出似然函数的下界，不断迭代，使得下界不断逼近$\mathcal{L}{(\theta)}$.


$$
\begin{eqnarray*}
\mathcal{L}{(\theta)} &=& \text{log}P(X|\theta) \tag{1}\\
&=& \text{log}P(X,Z|\theta) - \text{log}P{(Z|X,\theta)} \tag{2}\\
&=& \text{log}\frac{P(X,Z|\theta)}{q(Z)} - \text{log}\frac{P{(Z|X,\theta)}}{q(Z)} \tag{3}
\end{eqnarray*}
$$
等式两边同时对$q(Z)$求期望：
$$
\begin{eqnarray*}
\text{left} &=& \int_{Z}q(Z)\text{log}P(X|\theta)\text{d}Z \\
&=& \text{log}P(X|\theta)\int_{Z}q(Z)\text{d}Z \\
&=& \text{log}P(X|\theta)
\end{eqnarray*}
$$

$$
\begin{eqnarray}
\text{right} &=& \int_{Z}q(Z)\text{log}\frac{P(X,Z|\theta)}{q(Z)}\text{d}Z - \int_{Z}q(Z)\text{log}\frac{P{(Z|X,\theta)}}{q(Z)}\text{d}Z \\

&=& \int_{Z}q(Z)\text{log}P(X,Z|\theta)\text{d}Z -\int_{Z}q(Z)\text{log}q(Z)\text{d}Z- \int_{Z}q(Z)\text{log}\frac{P{(Z|X,\theta)}}{q(Z)}\text{d}Z \\

&=& \underbrace { \int_{Z}q(Z)\text{log}P(X,Z|\theta)\text{d}Z -\int_{Z}q(Z)\text{log}q(Z)\text{d}Z }_{ ELBO } 
+ KL\left(q(Z)||P(Z|X,\theta)\right)
\end{eqnarray}
$$

所以：
$$
\text{log}P(X|\theta) = \underbrace { \int_{Z}q(Z)\text{log}P(X,Z|\theta)\text{d}Z -\int_{Z}q(Z)\text{log}q(Z)\text{d}Z }_{ ELBO } 
+ KL\left(q(Z)||P(Z|X,\theta)\right) \tag{4}
$$
上式中，$ELBO(\text{evidence lower bound})$是一个下界，所以$\text{log}P(X|\theta) \geq ELBO$，当$KL$散度为0时，等式成立。

也就是说，不断最大化$ELBO$等价于最大化似然函数。在EM迭代过程中的第$i$ 步，假设$q(Z)=q(Z|X,\theta^{(i)})$，然后最大化$ELBO$
$$
\begin{eqnarray}
\hat{\theta}^{(i+1)} &=& \underset{\theta}{\text{argmax}}ELBO \\
&=& \underset{\theta}{\text{argmax}} \int_{Z}q(Z|X,\theta^{(i)})\text{log}P(X,Z|\theta)\text{d}Z -\underbrace{\int_{Z}q(Z|X,\theta^{(i)})\text{log}q(Z|X,\theta^{(i)})\text{d}Z}_{\text{independent with } \theta}  \\
&=& \color{red}{\underset{\theta}{\text{argmax}} \int_{Z}q(Z|X,\theta^{(i)})\text{log}P(X,Z|\theta)\text{d}Z} \tag{5}
\end{eqnarray}
$$


## 4. EM公式导出之ELBO+Jensen Inequality

### 4.1 Jensen Inequality



### 4.2 EM公式推导

对log-likelihood做如下变换：
$$
\begin{eqnarray*}
\text{log}P(X|\theta) &=& \text{log}\int_{Z}P(X,Z|\theta)\text{d}Z = \text{log}\int_{Z} q(Z) \frac{P(X,Z|\theta)}{q(Z)}\text{d}Z \\
&=& \text{log}\mathbb{E}_{q(Z)}\left(\frac{P(X,Z|\theta)}{q(Z)} \right) \\
&\geq& \mathbb{E}_{q(Z)} \left[\text{log}\frac{P(X,Z|\theta)}{q(Z)}  \right] \\
&=& ELBO

\end{eqnarray*}
$$
只有当$P(X,Z|\theta) = C \cdot q(Z)$时，等号才成立。



## 5. EM收敛性证明

如果能证明
$$
P(X|\theta^{(i+1)}) \geq P(X|\theta^{(i)})
$$
则说明EM是收敛的，因为$P(X|\theta)$肯定有界，单调有界函数必收敛！



$$
\begin{eqnarray*}
\text{log}P(X|\theta) &=& \text{log}P(X,Z|\theta) - \text{log}P{(Z|X,\theta)} \\
&=& \underbrace{\int_{Z}p(Z|X,\theta^{(i)}) \text{log}P(X,Z|\theta) \text{d}Z}_{Q(\theta,\theta^{(i)})} - \underbrace{\int_{Z}p(Z|X,\theta^{(i)}) \text{log}P{(Z|X,\theta)}\text{d}Z}_{H(\theta,\theta^{(i)})}
\end{eqnarray*}
$$
由于$\theta^{(i+1)}$使得$Q(\theta,\theta^{(i)})$达到极大，所以：
$$
Q(\theta^{(i+1)},\theta^{(i)}) - Q(\theta^{(i)},\theta^{(i)})\geq 0
$$

$$
\begin{eqnarray*}
H(\theta^{(i+1)},\theta^{(i)}) - H(\theta^{(i)},\theta^{(i)}) &=& \int_{Z}p(Z|X,\theta^{(i)}) \text{log}P{(Z|X,\theta^{(i+1)})}\text{d}Z - \int_{Z}p(Z|X,\theta^{(i)}) \text{log}P{(Z|X,\theta^{(i)})}\text{d}Z \\

&=& \int_{Z}p(Z|X,\theta^{(i)}) \frac{\text{log}P{(Z|X,\theta^{(i+1)})}}{\text{log}P{(Z|X,\theta^{(i)})}}\text{d}Z \\

&=& -KL\left(p(Z|X,\theta^{(i)}) || \text{log}P{(Z|X,\theta^{(i+1)})} \right) \\

&\leq& 0


\end{eqnarray*}
$$

因此，得到：
$$
P(X|\theta^{(i+1)}) \geq P(X|\theta^{(i)})
$$




