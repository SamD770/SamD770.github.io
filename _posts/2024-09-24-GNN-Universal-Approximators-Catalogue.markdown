---
layout: post
title:  "A catalogue of Universal Approximators on Graph-Structured data"
---

# Cataloguing Universal Approximators on Graph-Structured Data 

GNNs can be proven to give the same output for certain pairs of discrete graphs [^xu_2018_wl_expressivity] [^morris_2018_wl_expressivity]. 
There's a litany of work which improves the expressivity of GNNs via adding more features, more hops to message passing etc, but  these enhanced GNNs often still have blind spots. 

It feels to me like these blind spots are not a big deal in terms of biochemistry applications: if a graph has any distinguishing node features at all the expressivity greatly increases. 
Nonetheless, a history of patching over a theoretical problem with quick fixes seems to indicate that the field may be yet to unlock a more fundamental breakthrough.

## Reading list:


structural message passing paper [^vingac_2020_structural_message]

what NNs cannot learn, touches on similarity[^loukas_2020_cannot_learn] 

equivalence between isomorphism and function approximation [^chen_2019_isomorphism_approximation]

Review paper: Weisfeiler and Leman go Machine Learning: The Story so far

## The papers

With this motivation, I want to catalogue prior techniques which don't have these blindspots.
There is a general pattern which these works use to show that their methods can distinguish any two distinct graphs which looks like:


1. Devise a scheme by which an internal representation of the graph's adjacency matrix can be acheived by the model.

2. Use universal approximation results for standard Neural Networks to show that the model can process its internal representation of the adjacency matrix to solve the graph isomorphism problem.

[This pattern of boostrapping universal approximation onto graphs may in fact be necessary for an efficient universal approximation algorithm, the graph isomorphism problem is thought to be quite hard, possibly NP-complete, and thus solving it explicitly using one's algorithm would lead to intractable numbers of computations]: #

**Node identification methods** such as random node initialisation [^abboud_2021_random_init] [^sato_2021_random_features]
relational pooling [^murphy_2019_relational_pooling] suggests that we do all permutations of the nodes as input to a neural network **TODO: read structural message passing paper [^vingac_2020_structural_message]**

**Spectral methods** [^kreuzer_2021_spectral_attention] Exphormer: Sparse Transformers for Graphs

**Asynchronous GNN methods** [^finkelshtein2024cooperative] Essentially use the randomness of the walking to give expressivity 

**Invariant/equivariant GNNs**  [^maron_2018_invariant_networks] [^maron_2019_provably_powerful] [^maron_2019_universality_invariant] [^keriven_2019_universal_invariant]

These approaches share similar downsides:

- The number of features needs to scale with the number of nodes in the graph. To deal with this, you need to pad the features of smaller graphs in your dataset. This makes me doubt that generalisation will be easy if your dataset spans many OOMs of node counts (eg. for a biochemistry foundation model).

- Injecting randomness into the initialisation of the method to some degree breaks symmetry (specifically, nodes in the same orbit will be assigned different node features). It is possible (although intractable) to have completely deterministic symmetry-preserving methods by averaging over all $\vert V \vert !$ initialisations, as proposed by [^murphy_2019_relational_pooling] **TODO: read [^yarotsky_2022_universal_invariant]** so at least in theory symmetry-breaking randomness isn't necessary for universal approximation.

#### Aside: Why even care about universal approximation anyway?

Universal approximation is the only big theoretical result concerning standard nns that we have after ~60 years of study. At first inspection, even universal approximation seems a bit underwhelming: the bounds on width/depth of the networks are exponential in the problem size, and thus essentially equate to the neural network being able to simulate a lookup table (This is in fact necessary, as most functions in $L^2(\R^{n})$ or $C(\R^{n})$ have no structure). 

Most of the problems we care about have some form of structure, in that they can be computed more efficiently by a process than a lookup table. We can exploit this with the fact that _composing neural networks itself gives a neural network_ to see that, not only is there a large lookup table nn solving the problem, but that there is a deeper nn that solves the problem via the process by learning the intermediate representations. 

Example: consider the problem of modular addition of pairs of one-hot encoded integers modulo $p$ (à la grokking []). Arbitrary-width universal approximation tells us that there exists a two-layer lookup table nn that takes any pair of integers and looks up the result. It also tells us that there exists a 6-layer nn that 1. maps the one-hot encodings to a circular representation 2. adds the circular representations 3. decodes the resulting circular representation.

In fact the lack of theoretical results in deep learning is probably due to the fact that we are constrained theoretically to considering unstructured function domains like $L^2(\R^{n})$ or $C(\R^{n})$, creating a description of the structure in the functions that we want to approximate is in itself the problem. 

#### Aside: Exploring restriction of labels

The latter point gives rise to an idea: do we actually need to give all nodes unique IDs to give universal approximation? 

Unfortunately, we see that even trying to give nodes in the same orbits (generated by the graph isomorphism group) the same label trivially fails because in cycle graphs every node is in the same orbit. 

What if, instead of using $\vert V \vert !$ permutations of all possible node labels we could instead use just $\vert V \vert$ labellings, considering the graph "rooted" at each node individually? [^you_2021_identity_aware] Propose doing this. 

TODO: there are several papers coming out of uni of Peking that claim that these ID-aware GNNs can't count cycles, I don't see how this is the case. 

#### Aside: Homomorphism expressivity

There's a string of papers using graph homomorphisms to quantify isomorphisms [^lovasz_1967_operations], [^dell_2018_lovasz_wl], [^zhang_2024_beyond_wl]. The proofs in these papers are quite dense, here I want to give an intuitive explanation of why they're correct

As a primer, a [graph homomorphism](https://en.wikipedia.org/wiki/Graph_homomorphism) is a structure-preserving mapping between two graphs $\phi: G \to F$. Specifically, it requires that for two connected nodes in $G$, $u - v$ we have $\phi(u) - \phi(v)$. Note that a homomorphism need not be _injective_, in that it may map two nodes to the same node or _surjective_, in that it map map two .


Lovász [^lovasz_1967_operations] works with [uniform hypergraphs](https://en.wikipedia.org/wiki/Hypergraph#Properties_of_hypergraphs), but in this blog post we'll stick to grpahs. 

Let $\Gamma$ be the set of all finite graphs. Lovász [^lovasz_1967_operations] shows that, for any graph $G \in \Gamma$, the $\Gamma$-indexed vector $\langle G \rangle$ defined by the number of homomorphisms count $\langle G \rangle _F = \# Hom(F, G)$ completely determines the isomorphism class of $G$. 


####  Aside: the connection between $FOC_2$ and the Weisfeiler-Lehman test. 

There's a cross-paper chain of reasoning connecting first-order logic to the WL-test to universal approximation that looks like:

[^cai_1992_optimal_variables] show that two graphs are WL-distinguishable iff their nodes are $FOC_2$ distinguishable.

[^barcelo_2020_logical_expressiveness] show that all $FOC_2$ sentences are expressible by sufficiently large GNNs.

[^abboud_2021_random_init] show that, for graphs with node IDs, graph isomorphism is an $FOC_2$ problem.

There seems to be a parrallel line of reasoning  that goes something to the effect of 

1. GNNs can approximate the WL test [^xu_2018_wl_expressivity]
2. The WL test with node ids can, in theory, solve graph isomorphism, with the condition that building a sufficiently large lookup table and hash function range to account for all $n!$ assignments of node ids is intractable in practice.


#### Aside: Understanding Equivariant Graph Networks with Einstein summation

["Point of view is worth eighty IQ points." ](https://quoteinvestigator.com/2018/05/29/pov/)- [Alan Kay](https://en.wikipedia.org/wiki/Alan_Kay)

Invariant/Equivariant networks [^maron_2018_invariant_networks] are a method of processing graph-structured data which don't appear to rely on the message-passing framework. Here I'm re-writing many of the seminal results in this field due to Maron et al.[^maron_2018_invariant_networks][^maron_2019_provably_powerful][^maron_2019_universality_invariant] using Einstein summation notation, as I found this particularly illumintating.

**notation:** Let $P$ be a permutation matrix, $\pi$ be the corresponding permutation. So that for the standard basis:

$$e_1, e_1 \dots e_n = (1, 0, \dots, 0)^T, (0, 1, \dots, 0)^T, \dots (0, 0, \dots, 1)^T.$$

We have $P e_i = e_{\pi(i)}.$

Let $P \star A$ be the result of permuting an adjacency matrix $A$'s rows and columns according to $\pi$. We see that:

$$(P \star A)_{ij} = (P A P^T)_{ij} = A_{\pi(i) \pi(j)}$$


**How do in/equi-variant networks work?**
The fundamental idea of in/equi-variant networks is to learn functions of matrices that are in/equi-variant to simultaneous perumatation of the rows and columns, and then apply this function to the adjacency matrix. This way, no matter the ordering of the nodes in the adjacency matrix representation of a graph the function takes the same value. I will focus on the more complex equivariant case in this post, once you understand it the invariant case becomes obvious. 

In Section 3 of [^maron_2018_invariant_networks] the authors consider the case of equivariant linear transformations of the adjacency matrix $L: \R^{n  \times n} \to \R^{n  \times n }$, ie those satisfying both: 

$$L(A +  \lambda B) = L(A) +  \lambda L(B)$$

and

$$L(P \star A) = P \star L(A)$$

Basic linear algebra tells us that the first condition makes $L$ representable as a $n^2 \times n^2$ matrix, ie :

$$(L A)_{ij} = {\color{lightgray}\sum_{s, t}} L_{ij}^{st} A_{st}$$

for some scalars $\{L_{ij}^{st}\}_{1 \leq i, j, s, t \leq n}$ (note that the superscripts here are indices, not exponents). I have coloured the summation sign in light grey as from now on it will be ommitted, as per the [einstein summation convention](https://en.wikipedia.org/wiki/Einstein_notation). Using this notation, the LHS and RHS of the equivariance definition become:

$$L(P \star A)_{ij} = L_{ij}^{st} (P \star A)_{st} = L_{ij}^{st} A_{\pi(s) \pi(t)},$$

$$(P \star L(A))_{ij} = L(A)_{\pi(i)\pi(j)} = L_{\pi(i)\pi(j)}^{s't'} A_{s't'}.$$

Where we have used $s', t'$ to avoid confusing summation indices. In order to get them on the same summation index, we can take $s', t' = \pi(s)\pi(t)$ and observe that the above must be true for all $A$, including those with only one non-zero entry to see that for all permutations $\pi$:

$$L_{ij}^{st} = L_{\pi(i)\pi(j)}^{\pi(s)\pi(t)},$$

which is equivalent to Equation (2) of [^maron_2018_invariant_networks] while avoiding tricky Kronecker products and matrix unstacking. We can also see that Proposition 1. of [^maron_2018_invariant_networks] will follow by noting that at no points we used the fact that there are only two indices in the above proofs, and thus we may simply replace i, j with $k$ indices $i_1, i_2, \dots i_k$ and $s, t$ with $s_1, s_k \dots s_k$ to generalise the above to $k$ th-order tensors (although I will stick to graphs for the rest of this post). 

Now, consider the case of $L_{00}^{00}$. We see that for any $i$, there exists a perumtation $\pi$ such that $\pi: 0 \mapsto i$, and thus:

$$L_{00}^{00} = L_{11}^{11} = \dots = L_{nn}^{nn}.$$

But what about arbitrary $L_{ij}^{st}$? In general, we start to construct a perumtation $\pi$ between two arbitrary tuples of indices $i, j, s, t$ and $i' j', s', t'$ by choosing $\pi(s) = s'$, $\pi(t) = t'$, etc. This will work if and only if the "equality pattern" of the indices is the same. 

For example, for any distinct indices $i \neq s$, we see that all the coefficients of the form $L_{i s}^{s i}$ must be equal.

Taking a step back, we can now visualise what is happening. In general, an equivariant linear map $L: \R^{n^2} \to \R^{n^2}$ takes a (directed) adjacency matrix $A$ and outputs a matrix $L(A)$ which assigns a value to every ordered pair of nodes. 


every ordered pair $i, j$ of nodes in our graph

consider an adjacency matrix $A$ and have nodes 

For example in the above we see that $(0, 0, 0, 0), (1, 1, 1, 1) \dots (n, n, n, n)$ all share the same equality pattern.

In reality, we want to handle node and edge features, to do this we simply let $A_{st} \in \R^d$ be a matrix and $L_{ij}^{st} \in \R^{d \times d'}$


**Universal approximation of invariant networks: polynomials route [^maron_2019_universality_invariant]**

[^maron_2019_provably_powerful]



#### Aside: Using padding to give a defintion of linearity for GNN layers (Taken from my master's dissertation).

- explanation of padding invariance
- explanation 

#### Aside: 3D geometric graphs are a special subset of graphs

The set of graphs which could appear in GNNs applied to biochemistry is restricted from the set of all graphs: if $u$ connects to $v$ then for all $w$ such that $w$ is closer to $u$ than $v$, ie $\lVert u - w \rVert^2 \leq \lVert u - v \rVert^2 $ we have that $u$ connects to $w$. Open question: can we rephrase this condition using eg. triangle inequality to be in purely graph theoretic language? 

- Look into Florian's stress loss
- Group in Vienna shows universality for planar graphs "Maximally Expressive GNNs for Outerplanar Graphs"