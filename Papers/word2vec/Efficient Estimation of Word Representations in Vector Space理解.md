# Efficient Estimation of Word Representations in Vector Space理解



## 1. Introduction

之前的nlp系统和技术将词看做是原子的，比如用于统计语言模型的N-gram模型。好处是简单、较好的鲁棒性和可视化。但也有弊端：

因为它是一个简单的模型，如果训练数据很多，效果还可以。但是对于特定领域的问题，可能受限于语料库的大小，模型达不到理想的效果。



所以，一些复杂的模型被提了出来，比如词的分布式表示，基于神经网络的语言模型等。



### 1.1 Goals of the Paper





