# NFNets
Introduction
============
One of the most important components that allows ResNets (or any Neural networks to be general) to be deeper is Batch Normalization. According to [1], it ensures good 
signal propagation which allows us to train deeper networkds, without convolution activations getting exploded. However, there are certain caveats with
Batch-Normalization. It breaks the independence between training examples, memory overhead and also poses difficulty in replicating the trained models 
on different hardware. \
To counter the above disadvantages, I explored NFNet - Normalizer-Free ResNets, for the project, to free typical 
ResNets from batch normalization for good. Not only they make the training faster but according to [2] even the smaller models
made of NFNets match the performance of EfficientNet (one of the SOTA models) on imagenet.

NFNets in Brief
==============
**Freeing the Normalization**: According to [4], in typical ResNets, Batch normalization downscales
the input to each residual block by a factor proportional to the standard deviation of the input signal.
And each residual block increase the variance of the signal by an almost constant factor. Keeping
these two findings in mind, authors in [3], proposes the following modified residual block that mimics
the above 2 findings. That is,
```math
x_{l+1} = x_{l} + \alpha f(\frac{x_{l}}{\beta_{l}})
```
where xl denotes the input to the lth residual block, $f_{l}(.)$ denotes the residual block function, α
denotes the hyperparameter (recommended value is 0.2) and $β_{l}$ is choosen to be $Var(x_{l})$. Initializing
the weights of the function $f$ such that $Var(f(x)) = Var(x)$, gives an analytical form to derive $β_{l}$. We
achieve initialization that preserves variance through He or Kaiming initialization.
With the above initialization, $β_{l}$ can be predicted with the following recurrence relation

```math
\beta_{l} = Var(x_{l}) = Var(x_{l-1}) + \alpha^{2}
```
Since we normalize the data, $Var(x_{o}) = 1$. The above mentioned modified residual block, along
with $α$ and $β_{l}$ will help in good propagation of signal without being exploded.

**Recifying activation induced mean shifts**: According to [3], it was observed that changing
the residual block form, although helped, did introduced few practical challenges that
arose from the mean shifts seen in hidden activations. To curb this mean shift and ensure
that the variance in the residual branches are preserved (from exploding), scaled weight
standardization, inspired from [5], is proposed by the authors in [3]. The authors suggest that
we re-parameterize the weights of the convolution layers through the training in forward pass as below
```math
\hat W_{i, j} = \gamma \frac{W_{i, j} - \mu_{i}}{\sigma_{W_{i}}\sqrt{N}}
```
Where $\mu$ and $\sigma$ are calculated across fan-in of the convolution filters. $\gamma$ is the scaling dependent
on the kind of activation the network uses. For my network I used ReLU as activations, for which $\gamma = \frac{\sqrt{2}}{\sqrt{1 - \frac{1}{\pi}}}$ (this value is derived in [3]).

References
==========
[1] [https://arxiv.org/pdf/2101.08692.pdf]
[2] [https://arxiv.org/pdf/2102.06171.pdf]
[3] [https://arxiv.org/pdf/2101.08692.pdf]
[4] [https://proceedings.neurips.cc/paper/2020/file/e6b738eca0e6792ba8a9cbcba6c1881d-
Paper.pdf]
[5] [https://arxiv.org/pdf/1903.10520.pdf]
