# 内部协方差偏移

在深度神经网络模型里面，我们通常会选择对各层的输出添加不同的激活函数，让模型能够加入非线性，学习更多维度的数据信息。

但是在对数据流动的过程添加激活函数后，激活输出会面临一个问题：internal convariate shift（ICS）.

内部协方差偏移是指**训练过程中深度网络内部节点分布的变化**。

**paper:[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)**

在这篇文章中指出模型节点产生的内部协方差偏移是影响模型收敛。

同时提出了著名的Batch normalization。BatchNorm的基本思想是在每个batch里面，对每一层的激活进行规范化处理，从而来降低ICS的影响，保证每层输出的数据分布的稳定。

$$
x_{i+1}=\gamma \frac{x_i-\mu_b}{\sigma_b}+\beta
$$

可以肯定的是，在神经网络中加入BatchNorm对于梯度和损失都有明显的改善。

**paper: [How Does Batch Normalization Help Optimization?](https://proceedings.neurips.cc/paper_files/paper/2018/file/905056c1ac1dad141560467e0a99e1cf-Paper.pdf)**

在这篇文章中提出，BatchNorm和ICS之间的没有联系，至少联系是非常微小的。甚至BatchNorm不能降低ICS。

为了实验验证，这篇文章在BatchNorm之后加入了随机噪音，以此来破坏输出的数据分布，实验证明，即使协方差偏移了，但是最后的模型性能依然比没有batchNorm和随机噪音的标准模型来的好。

文章指出BatchNorm能使loss landscape更加平滑。也就是说，**损失以较小的速率变化，梯度的大小也较小**。这个也容易解释，因为Batch Norm能够将数据压缩到一个小的范围内，并且能够使大部分的数据离开激活函数的饱和范围（saturation regions），那么每一次的训练迭代中得到的损失值也不会过大，并且这些损失值差距也不会很大。梯度也随之变小并且尽量避免梯度消失和梯度爆炸。



**在这里不过多描述ICS，因为目前为止，并不认为ICSBatchNorm的根本原因。现在所用的技术，在某种程度上，是超过理论可解释的，也许在将来被更合理得解释后，可再做更新。**



此外，这里还有一篇文章：

[Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift](https://arxiv.org/pdf/1801.05134.pdf)

探讨了将Dropout和BatchNorm组合使用导致性能变差的原因，并提出了能缓解这些问题的策略。



## Reference

1. https://medium.com/analytics-vidhya/internal-covariate-shift-an-overview-of-how-to-speed-up-neural-network-training-3e2a3dcdd5cc
2. https://arxiv.org/pdf/1502.03167.pdf