---
layout: post
title:  "Per-Sample Gradients Show What Your Generative Model Knows"
---

This blog post is to accompany our paper,["Approximations to the Fisher Information Metric of Deep Generative Models for Out-Of-Distribution Detection"](https://openreview.net/forum?id=EcuwtinFs9&), which has been accepted to TMLR. 

Our code is available on [github](https://github.com/SamD770/Generative-Models-Knowledge). 

### Motivation

Recognsing novelty is a fundamental element of cognition. With novelty recognition, we can be curious about new things, direct our attention to them, and learn from them efficiently. For those interested in AI, a natural question is thus: how can we model novelty in the information theoretic framework which underpins deep learning?

The solution may at first seem obvious: use the model likelihood! [^bishop](## "hover text") 

In deep learning, we start with a dataset $\bf{x}_1 \dots \bf{x}_N$ of images which we 

- Example: Do deep generative models know what they don't know?

#### Primer: why isn't the likelihood a good measure for OOD detection?

- Example: discrete data, bit strings

- Do deep generative models know what they don't know?


![Poster from Nalisnick et al](assets\typicality-poster.png)*From Nalisnick et al.'s [poster](https://www.gatsby.ucl.ac.uk/~balaji/BDL-NeurIPS2019-typicality-poster.pdf)*

So almost all the samples from a distribution can exist inside some small band of entropy values, and there can still exist very low volume regions with lower entropy than this band of values. But is the converse true? Does a sample being in this small band of entropy values mean it has the same semantic properties as the training data? Unfortunately not. 


- Example: typicality can't explain everything 

- Example: continuous data, RGB-HSV transformation

- Likelihood ratios

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

- Layman's explanation of FIM results, normal distribution of $L^2$ norm results

- Layman's explanation of our method

#### Beyond OOD detection

- Anthropic's empirical influence functions paper

#### Interesting future research directions

- "OOD detection" isn't well defined, does the function of a layer correspond to the type of OOD detection it signals

- Using eg. the Sherman-Morrison formula to compute unbiased MC estimates for the FIM

- Can the FIM-induced inner product from small LMs be used for batching federated learning in LLMs?

- Adam : Our method :: Sharpness aware minimization : ?

- Cramer-Rao bound near optima of DGMs(?)

#### Bibtex


TODO: TMLR paper BibTex

Before TMLR, Workshop paper BibTex

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

[^bishop]: This is the first footnote.
