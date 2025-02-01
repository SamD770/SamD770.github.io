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

This blog post is to complement our paper,["Approximations to the Fisher Information Metric of Deep Generative Models for Out-Of-Distribution Detection"](https://openreview.net/forum?id=EcuwtinFs9&), which has been accepted to TMLR. 

<!-- It is intended to be:

- A position paper on OOD/novelty detection as a whole.
- A tutorial on the counter-intuitive stats/information theory of high-dimensional models on continuous data.
- If you want a detailed description of our methodology and results, please go to our paper. -->

[Our code](https://github.com/SamD770/Generative-Models-Knowledge) is available on github. 

### Motivation

Recognising novelty is a fundamental element of cognition. With novelty recognition, we can be curious about new things, direct our attention to them, and learn from them efficiently. 
For those interested in interpretable AI, a natural question is thus: how can we measure novelty in the information theoretic framework which underpins deep learning?

#### Primer: the model likelihood as a measure of novelty?

The solution may at first seem obvious: the model likelihood! [^bishop] To be more specific, many generative models $p^{\theta}$ on images, text etc. can both generate new samples ${\bf x} \sim p^{\theta}$ as well as evaluate (perhaps an approximation of) the log-likelihood of a given sample $\log p^{\theta}({\bf x})$. As these models are trained to maximise the log-likelihood on some training dataset, and at test time they generate samples semantically similar to these training data, it stands to reason that they should assign higher likelihoods to data which are semantically similar to their training data, and lower likelihoods to novel data.

Unfortunately, this doesn't work in practice. In their colourfully named seminal works "Do Deep Generative Models Know What They Don't Know?" and "WAIC, but Why?", Nalisnick et al. [^Nalisnick] and Choi et al.[^choi] concurrently demonstrated this for three types of generative image models (VAEs, normalising flows and autoregressive models), noting that models $p^{\theta}$ trained on CIFAR-10 (see the below samples) will assign log-likelihoods of $- \log_2 p^{\theta}({\bf x}) \approx 7,000$ for samples from SVHN, whereas they will assign log-likelihoods of $- \log_2 p^{\theta}({\bf x}) \approx 10,000$ for samples from CIFAR-10. Taking exponentials, this shows the completely novel dataset, SVHN, is being assigned an astronomically higher likelihood!  

<center>
<img src="\assets\svhn_samples.png" width="300" height="300">
<img src="\assets\cifar10_samples.png" width="300" height="300">
</center>

*<center><font color="gray">
The SVHN (left) and CIFAR-10 (right) datasets
(Source: <a href="http://ufldl.stanford.edu/housenumbers/" target="_blank">SVHN</a> and <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">CIFAR-10</a> releases)
</font></center>*

Why might this happen? The phenomenon occurs across models (including, we show, for diffusion models). 
These models generate samples CIFAR-10-like samples and give similar log-likelihoods on their training and test data, so explanations of poor models or overtraining don't have explanatory power.

In fact, the most simple example of this phenomenon can be found in classical probability. Consider generating samples of $10,000$-length strings of the form $THTHTHH \dots HTTTHH$ by flipping a coin that is $60\%$ biased towards heads. The most likely strings are all or almost-all heads, even though $\approx 99.99 \%$ of the strings we draw from the distribution will have a number of heads in the range $(5800, 6200)$. There is no law saying that, allowing the sampling distribution $q$ to vary, a distribution must maximise its own expected log-likelihood $\mathbb{E}_{\bf x \sim q} \log p({\bf x})$ (conversely, the case whereby the evaluation distribution $p$ varies [does hold](https://en.wikipedia.org/wiki/Gibbs'_inequality)).

So what is the property of SVHN that corresponds to having a large number of heads? Empirically, generative image models are inductively biased to assign higher likelihoods to images which are smoother or simpler images, with a constant image being assigned the highest likelihood. In fact, the implicit likelihood from the PNG compression algorithm correlates very strongly with that from a deep generative model [^serra].

In their follow up work [^typicality] Nalisnick et al. use this thought experiment with a high-dimensional gaussian to motivate a "typicality test", whereby we say that a sample is novel if its likelihood is too low *or* too high:

![Nalisnick et al.'s poster](\assets\typicality-poster-new.png)
*<center><font color="gray">
An intuitive explanation of Nalisnick et al.'s motivation for using typicality, 
(Source: 
<a href="https://www.gatsby.ucl.ac.uk/~balaji/BDL-NeurIPS2019-typicality-poster.pdf" target="_blank">Nalisnick et al.'s  poster</a>)
</font></center>*

So almost all the samples from a distribution can exist inside some band of *typical* log-likelihood values, and there can still exist very low volume regions with higher likelihoods than this band of values. But is the converse true? Does a sample being in this small band of typical values mean it has the same semantic properties as the training data? 

Unfortunately not. Repeating their original experiment but this time evaluating on a dataset of celebrity faces ([CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) instead of SVHN, Nalisnick et al.report [^typicality] that almost all of these celebrity faces images are contained in the band of likelihood which CIFAR-10 inhabits. Revisiting our example of coin-flip strings, a string with a block of 60 heads followed sequentially by 40 tails would be in the same band of likelihood values as most samples, but has a dissimilar structure.

Up until now, I have used the term "novelty" rather than more popular "out-of-distribution" (which we also use in our paper). This is because I think that the term "novelty" more intuitively captures the notion of a task which is both distinct from classification and possible. 
To elucidate what I mean here, consider these results:

If we pre-specify a certain out-distribution $q$ that we are interested in discriminating against, then the most powerful test for discriminating against $q$ [will be given by the likelihood ratio](https://en.wikipedia.org/wiki/Neyman%E2%80%93Pearson_lemma) $\log p^{\theta}({\bf x}) - \log q({\bf x}) $[^ren]. However, with $q$ well-specified our task becomes very close to classification. Furthermore, trying to discriminate against _any_ possible out-distribution which is inequal to the model $q \neq p^{\theta}$ will fail due to $q$s which are in a sense close to $p^{\theta}$ [^zhang] (a simple example of this for $p^{\theta}$ trained on all of CIFAR-10 would be to adversarially choose a single class of CIFAR-10 for $q$).

To avoid these paradoxes, we need to slightly break from the distributional paradigm by:
- Re-defining our problem as partitioning the data space into samples which are semantically similar to the training data and those that aren't (for now, we can simply consider distinguishing pairs of semantically dissimilar image distributions).
- Explicitly using the fact our model is a learning system instead of a fixed distribution. In our example of a block of $60$ heads and a block of $40$ tails, no model constrained to always interpreting the coin tosses as unordered or i.i.d. will be able to capture the novel structure here, whereas a model which has learned itself that the training dataset exhibits this structure might be able to *if* we allow it to explore outside its learned parameters.

#### Plan from here:

Gradients for anomaly detection
- Simple motivation for using learn-ability as a method of measuring novelty (brainstorm this)
- Derivation of gradients from a general "can you update to improve your model of the world"

Theory: natural Gradients
- High-level introduction to natural gradients, cite amari mostly, don't try to re-define everything
- Using FIM on normal distribution _including variance term_ recovers something typicality-esque 

Theory: representation dependence
- write up Normalising flow example

Future research directions
- Emphasise interpretability angle

#### Beyond the likelihood: gradients for novelty detection

Bearing the above in mind, we can come to a definition of novelty that extends beyond how probable a sample is: we can say that a sample contains novelty if it contains unlearned, but learnable, structure.

Consider the following scenario: while driving in your car, you observe a cloud shape that you haven't seen before. 
This exact configuration of the cloud is highly unlikely under your world model, yet you don't stop to learn the exact structure of the cloud, as doing so doesn't teach you about future clouds.
All the unpredictable structure of the cloud is at least for a human observer, unlearnable.
In contrast, consider observing a skateboarder on the road for the first time. There's learnable information about how the skateboarder

- want an example that manifestly equates likelihoods yet one is obviously novel and the other is obviously not

- a car version that you have seen before -> not novel as you don't update much
- a car version that you have not seen before -> novel, you can learn it and add to your mental dictionary

Consider your reaction to discovering the outcome of two events: 1. a coin flip vs 2. a very tight election.
Both of the events have outcomes with roughly $p = 0.5$ in your world model, but the discovering the binary outcome of the election is much more _interesting_, it provides a rich trove of information for your world model as to how voter opinions have drifted in comparison to polling, perceived likability of the candidates, et cetera. 
This information could be used to update your world model to make your predictions of other election outcomes better, demonstrating how the election outcome contains unlearned, but learnable, information. 

Note here that the definition is not fully complete, the election outcome is also _interesting_ because it updates your world model as to what the future will be. 
For example, if a country were decide to flip a coin to decide its leader. Is this novel?

Consider this: you're GPT-3, mid training run. 

First, you view 10 "1" and "0" characters from some plaintext representation of an encrypted file. it's leaked into your training data, perhaps someone posted it on reddit. Viewing 10 tokens from this string is essentially like viewing 10 Coin flips: the entropy is irreducible, it's not "novel" to you even if it contains 10 bits of information.

The string "453 + 362 = " appears to you. You can't quite yet do addition, you know that it has something to do with outputting digits, so all you can do is output a uniform distribution of the digits 0-9. However, the $\approx 10$ bits of information given in the answer "818" 

---

In general, for a model $M$ and a learning algorithm $step$, evalutation function $eval$ we define novelty of a sample to be:

$$
  Novelty(x, M, step) = eval(M, x) - eval(step(M, x), x)
$$

Why use $eval(M, x) = -\log p^M(x)$ instead of eg. $p^M(x)$ or $p^M(x)^2$?

If $eval$ and $step$ work together, we need that $eval(step(M, x), x) \geq eval(M, x)$. In general, we cannot expect for $eval(step(M, x), x) = eval(M, x)$ for all $x \sim p^M$ as repetitively training on one sample can (and should) lead to model collapse. 

We would ideally have that our novelty algorithm is probably approximately correct, ie that:

$$
\mathbb{P}_{x \sim P^{M}}(Novelty(x, M, step) < \epsilon_1) > 1 - \epsilon_2
$$

However, for a batch-based training algorithm we can guarantee that, in the limit for large batch sizes, we have $Novelty(x, M, step) < \epsilon$:

$$

$$

We wish for $eval(M, x)$ to be, in expectation, unhackable, such that if $M$ is the true model 

$$
  \mathbb{E}_{x \sim M'} \left{ \eval(M, x) \right}
$$

In terms of SGD:

\[
  step(M^{\theta}, x) 
  = 
  M^{\theta + \eta \frac{\partial}{\partial \theta}}
\]

And thus

\[
  eval(step(M^{\theta}, x), x)
  = 
  eval(M^{\theta + \eta \frac{\partial}{\partial \theta}}, x)
  = 
  \log p (x)
\]

---

Deep learning models are trained using gradients, so examining the gradient gives us a natural way to measure how much learnable structure there is in a sample.
For samples with structure that is already learned or unlearnable, we would expect the magnitude of the _gradient_ of the likelihood to be small, as updating the model's parameters with this gradient would not influence how well the sample's structure can be predicted.
We now have two problems:
- what, specifically, do we take the gradient of? (eg, why the log-likelihood and not the likelihood?)
- How do we measure the size of the gradient in a canonical way? 

To convince a skeptical reader that the gradient of the log-likelihood is a natural candidate to measure this, we consider two parts of theory: one from Amari, and one from myself.

#### Theory Primer: the Fisher Information Metric

- Citation for Amari

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

- Intuition of our method: learnability

#### Primer: representation dependence in continuous data

There is a frequently overlooked theoretical distinction between the discrete probability models we use for text and the continuous probability models we use for images.  For continuous data, the likelihood depends heavily on the notion of volume, with the likelihood evaluated at a point ${\bf x}$ [being defined as](https://en.wikipedia.org/wiki/Radon%E2%80%93Nikodym_theorem) the limiting ratio between the probability of a new sample being in some small ball-shaped neighbourhood $B$ of ${\bf x}$ and its volume, denoted $\mu(B)$: 

$$p({\bf x}) = \lim_{\mu(B) \to 0} \frac{\mathbb{P}(B)}{\mu(B)} \quad \text{where } {\bf x} \in B$$

For image models the volume term $\mu(B)$ is [implicitly defined](https://en.wikipedia.org/wiki/Lebesgue_measure) as the volumes of sets in the RGB input space $[0, 1]^{3 \times W \times H}$ , for example the volume of the set of all images with RGB values in the range [0.1, 0.9] is a high-dimensional box with volume  $0.8^{3 \times W \times H}$. To motivate our results, in our paper we show all MNIST images can be contained in a very low volume set of bounded smoothness. 

As Le Lan & Dinh [^lelan] argue, the volume term $\mu(B)$ has serious knock-on effects for the interpretability of likelihoods in generative image models. Consider the case of changing how we represent our input data (for example, changing the colour model from RGB to HSV). In abstract maths, this is an invertible transformation between two spaces $T: \mathcal{X} \to \mathcal{T}$ :

<center><img src="\assets\representation-dependence-explainer.png"></center>

*<center><font color="gray">
For continuous data, the likelihood depends on your choice of representation
(Source: 
<a href="https://www.geogebra.org/geometry/mruy2kfh" target="_blank">GeoGebra</a>)
</font></center>*

We see that $T$ can warp and squash sets in $\mathcal{X}$ when they are mapped to $\mathcal{T}$, changing their volumes. This means that the likelihood of some ${\bf x} \in \mathcal{X}$ will be different to the likelihood of the same point under the same distribution but with a different representation $p_{\mathcal{X}}({\bf x}) \neq p_{\mathcal{T}}(T({\bf x}))$.

Le Lan & Dinh [^lelan] show that, by considering arbitrary changes in representation $T$, the likelihoods of points can be arbitrarily re-ordered. 

As an example for image data, consider the case of changing the colour model from RGB to HSV. This change in representation spreads darker colours out into a large volume of HSV-space, causing images with darker patches to have much larger likelihood values under RGB than HSV. The resulting change in likelihoods for a given image is independent of the probability distribution $p$, and varies wildly for natural images. See below an example of the change in bits-per-dimension (BPD) from transitioning from an RGB to an HSV colour model for the first two samples in CIFAR-10:

$$
\Delta^{RGB \to HSV}_{BPD} = \frac{\log_2 p_{RGB}(\mathbf{x}) - \log_2 p_{HSV}(\mathbf{x})} {3 \times 32 \times 32},
$$

<center><img src="\assets\RGB-HSV-cifar-cropped.png"></center>

*<center><font color="gray">
Even simply changing the colour model while keeping the distribution fixed has a large effect on the likelihood.
(Source: <a href="https://openreview.net/forum?id=EcuwtinFs9&" target="_blank">our paper</a>)
</font></center>*

Using their observations, Le Lan & Dinh [^lelan] argue that a consistent definition of anomalies should satisfy a principle of invariance: given infinite training data, changing the representation shouldn't change which datapoints are classified as anomalies, a principle which typicality and likelihood thresholding violate.

In practice, we don't have infinte data and so want to use our choice of representation as an inductive bias. Does this invalidate Le Lan's principle of invariance for practicle use? 
I would argue that the principle of invariance *to volumes* is still justifiable for finite amounts of high-dimensional data.  
My mantra is that *distances induce good biases* whereas *volumes induce bad biases*: distances sum over dimensions of the data, whereas volumes multiply into exponential explosion/vanishing. 
Abstractly, the concepts of distance and volume are theoretically separate, in the general case a [space can have distances without this implying volumes](https://en.wikipedia.org/wiki/Metric_space), and [vice versa](https://en.wikipedia.org/wiki/Measure_space).

But given we train our model using the log-likelihood, which depends on the volume, isn't the volume already heavily baked in?
In our paper, we present the result that the gradient of the log-likelihood model is invariant to the choice of volume, the simple algeraic intuition being that taking logs and then derivatives separates and then nullifies the volume term:

$$
\nabla_{\theta} \log p^{\theta}({\bf}{x})
=
\lim_{\mu(B) \to 0} \nabla_{\theta} \log \mathbb{P}^{\theta}(B)
\quad \text{where } {\bf x} \in B
$$

For most generative models, this means that the entire initialisation, training, and sampling processes can be defined without ever implicitly assigning volumes to the input space.

**Example: a new way to compute gradients for your normalising flow** Normalising flows, and their continuous cousins [^neural_odes], learn an invertible mapping between...

We can compute the derivative as:

$$
\frac{\partial}{\partial \theta_i}  \log  \left\vert \det \frac{\partial f^{-1}_\theta}{\partial x} \right\vert = \nabla_z \cdot \left( \frac{\partial f^{-1}}{\partial \theta_i} \circ f\right)
$$

- sample $x'$ from 
- We can sample some vertices $x_1 \dots x_d$ as $x' + noise$
- We can compute $z'$ and $z_1 \dots z_d$ via passing backwards through $f^{-1}$
- We can integrate to find the probability of the $V$ enclosed by the $d+1$ vertices $z', z_1, z_d$
- We can compute the gradient of the integral:
- For each face of $V$ we can compute the flux induced by changing $\theta_i$, dot product with the gradient of $p(z)$ dz

Consider the case of a denoising diffusion probabilistic models: the input normalisation TODO: stuck here,

#### Interesting future research directions

As I've started my PhD I've changed tack with my research. Below are some ideas that I find interesting for future research in this topic, feel free to run with them:

- Empirical influence functions essentially take the FIM-induced dot-product between two different gradient vectors instead of the FIM-induced norm as we do. 
First suggested by Koh et al. [^koh], this gives a metric of how influential a training sample is in generating a given outcome.
Recently, Anthropic [^grosse] demonstrated some very interesting (if slightly worrying) results about how an LLM decides to request to not be shutdown based on the script of Kubrick's _2001: A Space Odyssey_, amongst other things.
There is definitely some interesting work to be done in relating these works by, for example, computing the FIM-induced norm on the train set to see which samples were overall most novel, at least when first exposed to the model.

- As we discuss, OOD detection isn't well-defined without specifiying some specific out-distribution one is interested in discriminating against. 
We know that different layers of neural networks are typically responsible for a heirachy of features, and thus it stands to reason that the FIM for a given layer would detect a different kind of out-distribution.

- We can use the [Sherman-Morrison](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula) formula allows for us to iteratively compute an estimate of the full-matrix FIM-induced inner product _only using the (tractable) inner product of gradient vectors_ via:

$$
F = \mathbb{E}_x\left[ s_x  s_x^\mathsf{T}\right] \approx F_N = \frac1{N + 1}(F_0 +  s_0  s_0^\mathsf{T} + \dots +  s_{N-1}  s_{N-1}^\mathsf{T})
$$

giving:

$$
s ((k + 1) F_k)^{-1} s^\mathsf{T} = 
s \left(k F_{k-1}+ s_{k-1}  s_{k-1}^\mathsf{T}\right)^{-1} s^\mathsf{T}
= s (k F_{k-1}^{-1}) s^\mathsf{T}- \frac {s (k F_{k-1})^{-1} s_{k-1} s_{k-1}^\mathsf{T} (k F_{k-1})^{-1}s^\mathsf{T}} {1 +  s_{k-1}^\mathsf{T}(k F_{k-1})^{-1} s_{k-1}}.
$$

- Do the FIM-induced norms of small models correlate with those of larger models? If so, can we apply insights from computing FIM-norms on small models to efficient curriculum/federated/active learning schemes on large models?

- By backpropagating twice, we could analyse not just which samples have the highest FIM-norm but which parts of these samples contribute most.

- Does the FIM-induced-norm of samples during training predict grokking behaviour?

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
url={https://openreview.net/forum?id=EcuwtinFs9}
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

[^choi]: WAIC, but why?

[^typicality]: typicality

[^serra]: input complexity

[^ren]: likelihood ratios

[^zhang]: understanding failures

[^koh]: Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions.
In International Conference on Machine Learning, pages 1885–1894. PMLR, 2017.

[^grosse]: Studying Large Language Model Generalization with Influence Functions

[^lelan]: Perfect Density Models Cannot Guarantee Anomaly Detection

[^neural_odes]: Neural Ordinary Differential Equations
