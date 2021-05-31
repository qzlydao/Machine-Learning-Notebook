[TOC]

## 摘要

对于有连续隐变量的有向图模型，当后验很难求或有大量数据时，该如何有效地推断和学习呢？

作者工作的贡献：

1. 对应变分下界，引入重参数技巧(reparameterization trick)，得到的下界近似函数可以直接用标准的随机梯度上升进行优化得到极值；
2. 可以有效计算后验的近似推断。



## 1. 引言

对于含有连续隐状态的有向图模型，隐状态的后验分布式很难计算的。

变分贝叶斯（VB）方式的做法是优化后验的近似函数得到后验的估计。但是，基于平均场理论，$ELBO$对近似后验分布的期望也是很难直接计算的！作者引入重参数化技巧，可以得到下界的可微无偏估计，从而可以使用标准的随机梯度上升法进行求解。



对于独立同分布的数据（且每个数据点都含有连续的隐变量），作者提出了自编码变分贝叶斯算法（AEVB）\



## 2. 方法

数据集样本之间是iid的，每个数据都有一个隐变量，对于全局参数，用MLE或MAP，对于隐变量，用VI。

<img src="Figure 1.directed graphical model.png" alt="Figure 1.directed graphical model" style="zoom:50%;" />



### 2.1 问题描述

数据集：$\mathbf{X}=\left\{ \mathbf{x}^{(i)} \right\}_{i=1}^{N}$  $N$个样本之间是i.i.d. 的，$\mathbf{x}$或连续或离散；

$\mathbf{z}$ : 连续随机隐变量



数据的生成过程可分为两步：

1. 依先验分布$p_{\theta^{*}}(\mathbf{z})$生成$\mathbf{z}^{(i)}$；
2. 依条件概率$p_{\theta^{*}}(\mathbf{x|z})$生成样本$\mathbf{x}^{(i)}$



这里不对边缘概率或后验做任何假设（暗指平均场理论），主要是想得到一个更通用的算法：

1. Intractability

   边缘概率： $p_{\theta}(\mathbf{x})=\int p_{\theta}(\mathbf{z})p_{\theta}(\mathbf{x|z})\text{d}\mathbf{z}$

   后验概率：$p_{\theta}(\mathbf{z|x})=p_{\theta}(\mathbf{z})p_{\theta}(\mathbf{x|z})/p_{\theta}(\mathbf{x})$

   都是intractable的（时间复杂度是指数级的，比如GMM模型中$p_{\theta}(\mathbf{x})$有$\text{log}\sum$，直接计算不可行），即使用了基于平均场的变分（MFVB）也是不可求解的（高维度求积分，无法直接求解）；

2. A large dataset

   数据太多，参数更新很慢。基于采样的方法也不可行。

   

