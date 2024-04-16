# Internal covariate shift

In deep neural network models, we often choose to add different activation functions to the outputs of various layers to introduce non-linearity, allowing the model to learn more dimensional data information.

However, after adding activation functions to the data flow process, the activated outputs will face an issue: **internal covariate shift (ICS)**.

ICS refers to **the changes in the distribution of internal nodes within a deep network during the training process**.

**paper:[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)**

In this paper, it is pointed out that the ICS generated within the nodes of the model affects the convergence of the model.

At the same time, the famous Batch Normalization (BatchNorm) is introduced. The basic idea of BatchNorm is to normalize the activations in each layer for each batch, thereby reducing the impact of ICS and ensuring the stability of the data distribution in the output of the layer.

It is certain that incorporating BatchNorm in neural networks significantly improves both gradients and loss.

**paper: [How Does Batch Normalization Help Optimization?](https://proceedings.neurips.cc/paper_files/paper/2018/file/905056c1ac1dad141560467e0a99e1cf-Paper.pdf)**

The paper proposes that there is no link, or most tenuous link between BatchNorm and ICS. It even suggests that BatchNorm does not reduce ICS.

To validate this experimentally, the paper introduces random noise after BatchNorm to disrupt the output data distribution. The experiments demonstrate that even with the ICS, the final model performance still surpasses that of the standard model without BatchNorm and random noise.

The paper points out that BatchNorm can **make the loss landscape smoother**. This means that **the loss changes at a smaller rate and the magnitude of gradients is also smaller**. This is easy to understand because BatchNorm can shrink data with a smaller range and can help most of the data move away from the **saturation regions** of the activation functions. As a result, the loss values obtained in each training iteration are not too large, and the discrepancies between these loss values are also minimized. Consequently, the gradients are reduced and help avoid issues like gradient vanishing and exploding. 

Additionally, there is another paper here:
[Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift](https://arxiv.org/pdf/1801.05134.pdf)
This paper discusses the reasons why combining Dropout and BatchNorm can lead to worsened performance and proposes strategies to mitigate these issues.

These are currently the two prevailing viewpoints. 



**Here, we won't elaborate too much on ICS, as it is not currently considered the fundamental reason for BatchNorm. The techniques used now are, to some extent, beyond theoretical explanation. Perhaps, with a more rational explanation in the future, updates can be made accordingly.**