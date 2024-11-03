---
layout: post
title:  "Per-Sample Gradients Show What Your Generative Model Knows"
---

[script to load mathjax, from https://zjuwhw.github.io/2017/06/04/MathJax.html]: #

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    processEscapes: true
  }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

This blog post is to accompany our paper,["Approximations to the Fisher Information Metric of Deep Generative Models for Out-Of-Distribution Detection"](https://openreview.net/forum?id=EcuwtinFs9&), which has been accepted to TMLR. It is:

- A position paper on OOD detection as a whole
- A tutorial on the counter-intuitive stats/information theory of high-dimensional models on continuous data
- A motivation to read our paper :)

Our code is available on [github](https://github.com/SamD770/Generative-Models-Knowledge). 

### Motivation

Recognising novelty is a fundamental element of cognition. With novelty recognition, we can be curious about new things, direct our attention to them, and learn from them efficiently. For those interested in AI, a natural question is thus: how can we measure novelty in the information theoretic framework which underpins deep learning?

#### Primer: the model likelihood as a measure of novelty?

The solution may at first seem obvious: use the model likelihood! [^bishop] To be more specific, almost all generative models $p^{\theta}$ on images, text etc. can both generate new samples ${\bf x} \sim p^{\theta}$ as well as evaluate (perhaps an approximation of) the log-likelihood of a given sample $\log p^{\theta}({\bf x})$. As these models are trained to maximise the log-likelihood on some training dataset, and at test time they generate samples semantically similar to these training data, it stands to reason that they should assign higher likelihoods to data which are semantically similar to their training data, and lower likelihoods to novel data.

Unfortunately, this doesn't work in practice. In the seminal work "Do Deep Generative Models Know What They Don't Know?" Nalisnick et al. [^Nalisnick] demonstrate this for three types of generative image models (VAEs, normalising flows and autoregressive models), noting that models $p^{\theta}$ trained on CIFAR-10 (see the below figure) will assign log-likelihoods of $- \log_2 p^{\theta}({\bf x}) \approx 7,000$ bits for samples ${\bf x}$ from SVHN, whereas they will assign log-likelihoods of $- \log_2 p^{\theta}({\bf x}) \approx 10,000$ bits for samples from CIFAR-10, corresponding to the completely novel dataset SVHN being assigned an astronomically higher likelihood!  

<center>
<img src="\assets\svhn_samples.png" width="300" height="300">
<img src="\assets\cifar10_samples.png" width="300" height="300">
</center>

*<center><font color="gray">
The SVHN (left) and CIFAR-10 (right) datasets
(Source: <a href="http://ufldl.stanford.edu/housenumbers/" target="_blank">SVHN</a> and <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">CIFAR-10</a> releases)
</font></center>*

Why might this happen? The phenomenon occurs across models (including, we show,for diffusion models), which generate samples that appear close to CIFAR-10 and give similar log-likelihoods on their training and test data, so the standard explanations of the models simply being poor or overtrained don't hold weight.

Looking for the most simple example of this phenomenon, we can go to classical probability. Consider the case of $10,000$-length strings of the form $THTHTHH \dots HTTTHH$ representing the the outcomes of independent coin flips that are $60\%$ biased towards heads. It is clear here that the most likely string is all heads, and in general a string with a larger number of heads will have a higher likelihood than one with fewer heads, even though $\approx 99.99 \%$ of the strings we draw from the distribution will have a number of heads in the range $(5800, 6200)$. Thus there is no law saying that, allowing our sampling distribution to vary, a distribution must maximise it's own log-likelihood (conversely, the case whereby the evaluation distribution varies [does hold](https://en.wikipedia.org/wiki/Gibbs'_inequality)).

What corresponds to having a large number of heads in the case of images? The image smoothness (TODO: find citation) or the implicit likelihood from an image compression algorithm [^serra], (TODO: find more citations here)

In their follow up work [^typicality] Nalisnick et al. use a similar thought experiment with a high-dimensional gaussian to motivate a "typicality test", whereby we say that a sample is novel if its likelihood is too low *or* too high:

![Nalisnick et al.'s poster](\assets\typicality-poster-new.png)
*<center><font color="gray">
An intuitive explanation of Nalisnick et al.'s motivation for using typicality, 
(Source: 
<a href="https://www.gatsby.ucl.ac.uk/~balaji/BDL-NeurIPS2019-typicality-poster.pdf" target="_blank">Nalisnick et al.'s  poster</a>)
</font></center>*

So almost all the samples from a distribution can exist inside some band of *typical* log-likelihood values, and there can still exist very low volume regions with higher likelihoods than this band of values. But is the converse true? Does a sample being in this small band of typical values mean it has the same semantic properties as the training data? 

Unfortunately not. Repeating their original experiment but this time evaluating on a dataset of celebrity faces ([CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) instead of SVHN, we see that almost all of these celebrity faces images are contained in the band of entropy which CIFAR-10 inhabits. Revisiting our example of coin-flip strings, a string with a block of 60 heads followed sequentially by 40 tails would be in the same band of likelihood values as most samples, but has a dissimilar structure.

TODO: is this discussion necessary? Up until now, I have referred to "novelty" rather than "out-of-distribution" as a descriptor for the samples we are interested in, although the latter is more commonly used when studying this field in practice

Of course, if we pre-specify a certain out-distribution $q$ that we are interested in discriminating against, then the most powerful test for discriminating against $q$ [will be given by the likelihood ratio](https://en.wikipedia.org/wiki/Neyman%E2%80%93Pearson_lemma) $\log p^{\theta}({\bf x}) - \log q({\bf x}) $[^ren]. However, with $q$ well-specified our task becomes very close to classification. Furthermore, trying to discriminate against any possible out-distribution which is inequal to the model $q \neq p^{\theta}$ will fail due to $q$s which are in a sense close to $p^{\theta}$ [^zhang] (a simple example of this for $p^{\theta}$ trained on all of CIFAR-10 would be to aversarially choose a single class of CIFAR-10 for $q$).

To avoid these paradoxes, we need to slightly break from the distributional paradigm by:
- Re-defining our problem as partitioning the data space into samples which are in a sense semantically similar to the training data and those that aren't (for now, we can simply consider pairs of semantically dissimilar image distributions)
- Explicitly using the fact our model is a learning system instead of a fixed distribution. In our example of a block of $60$ heads and a block of $40$ tails, no model constrained to always interpreting the coin tosses as unordered or i.i.d. will be able to capture the novelty here, whereas a model which has learned itself that the training dataset exhibits this structure might be able to *if* we allow it to explore outside its learned parameters.

#### Primer: representation dependence

- Example: continuous data, 

<center><img src="\assets\representation-dependence-explainer.png"></center>

*<center><font color="gray">
For continuous data, the likelihood depends on your choice of representation
(Source: 
<a href="https://www.geogebra.org/geometry/mruy2kfh" target="_blank">GeoGebra</a>)
</font></center>*


- RGB-HSV transformation

<center><img src="\assets\RGB-HSV-cifar-cropped.png"></center>

*<center><font color="gray">
Even simply changing the colour model has a large effect on the cross-entropy, 
(Source: <a href="https://openreview.net/forum?id=EcuwtinFs9&" target="_blank">our paper</a>)
</font></center>*

#### Primer: the Fisher Information Metric

- Motivation for FIM: Given two distributions, what's the difference between them?

Many different possible definitions
Let's look at all those which differentiate

- All limits lead to the FIM

- Deriving typicality & hyperbolic geometry from the normal distribution's  FIM:

Consider the case of a normal distribution, parameterised by it's mean and variance $(\mu, \nu)$

$$
\log p^{\mu, \nu}(x) =  - \log (2 \pi \nu) -\frac{(x - \mu)^2}{2 \nu}
$$

we see that the derivatives are thus:

$$
\frac{\partial}{\partial \mu} \log p^{\mu, \nu}(x) = \frac{x - \mu}{\nu}
$$

$$
\frac{\partial}{\partial \nu} \log p^{\mu, \nu}(x) = -\frac{1}{\nu} + \frac{(x - \mu)^2}{2 \nu^2}
$$

The second partial derivatives are thus:

$$
\frac{\partial^2}{\partial \mu^2} \log p^{\mu, \nu}(x) = -\frac{1}{\nu}
$$
$$
\frac{\partial^2}{\partial \mu \partial \nu} \log p^{\mu, \nu}(x) = \frac{\mu - x}{\nu^2}
$$
$$
\frac{\partial^2}{\partial \nu^2} \log p^{\mu, \nu}(x) = \frac{1}{\nu^2} - \frac{(x - \mu)^2}{\nu^3}
$$

Taking expectations gives the Fisher Information:


$$
F^{\mu, \nu}(x) = 
\begin{pmatrix}
\frac{1}{\nu} & 0 \\
0 & \frac{1}{2 \nu^2}
\end{pmatrix}
$$
- FIM is deep

#### How to approximate the Fisher information metric of your deep generative model

<center><img src="\assets\FIM-windows.png"></center>

*<center><font color="gray">
The FIM of generative models can be approximated as diagonal, 
(Source: 
<a href="https://openreview.net/forum?id=EcuwtinFs9&" target="_blank">our paper</a>)
</font></center>*


- Layman's explanation of FIM results, normal distribution of $L^2$ norm results

- Layman's explanation of our method

#### Interesting future research directions

- Anthropic's empirical influence functions paper

- "OOD detection" isn't well defined, does the function of a layer correspond to the type of OOD detection it signals

- Using eg. the Sherman-Morrison formula to compute unbiased MC estimates for the FIM

- Can the FIM-induced inner product from small LMs be used for batching federated learning in LLMs?

- Adam : Our method :: Sharpness aware minimization : ?

- Cramer-Rao bound near optima of DGMs(?)

#### Bibtex


TMLR paper BibTex

```
@article{
dauncey2024approximations,
title={Approximations to the Fisher Information Metric of Deep Generative Models for Out-Of-Distribution Detection},
author={Sam Dauncey and Christopher C. Holmes and Christopher Williams and Fabian Falck},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=EcuwtinFs9},
note={}
}
```

ICLR Workshop paper BibTex

```
@inproceedings{gradients2023anomaly,
  title={On Gradients of Deep Generative Models for Representation-Invariant Anomaly Detection},
  author={Sam Dauncey and Christopher C. Holmes and Christopher Williams and Fabian Falck},
  booktitle={ICLR 2023 Workshop on Pitfalls of limited data and computation for Trustworthy ML},
  url={https://openreview.net/forum?id=deYF9kVmIX},
  year={2023}
}
```

TODO: blogpost BibTex (?)

#### References

[^bishop]: Bishop citation.

[^Nalisnick]: Do deep generative models know what they don't know?

[^typicality]: Typicality

[^serra]: input complexity

[^ren]: likelihood ratios

[^zhang]: understanding failures