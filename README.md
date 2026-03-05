# Computing Wasserstein Barycenters via Linear Programming

This repository contains the Python implementation accompanying the paper:

> **Computing Wasserstein Barycenters via Linear Programming**  
> Gennaro Auricchio, Federico Bassetti, Stefano Gualandi, Marco Veneroni
> CPAIOR 2019

## Overview

The code computes **Kantorovich–Wasserstein (KW) barycenters** of a set of discrete probability measures (e.g., normalised grey-scale images) by solving a single Linear Programming (LP) problem — an uncapacitated minimum-cost flow problem on a structured graph whose topology depends on the chosen ground metric.

---

## Problem Formulation

### Notation

Following the notation of the paper, let:

- $V = \{1,\ldots,N\}^2$ be the regular $N\times N$ pixel grid, with $n = N^2$ nodes.
- $K$ be the number of input images.
- $\mu_k \in \mathbb{R}^n_+$ be the $k$-th input discrete probability measure (normalised histogram), for $k = 1,\ldots,K$, so that $\sum_{i \in V} \mu_k(i) = 1$.
- $G = (V, A)$ be a directed graph on the pixel grid, with arc set $A \subseteq V \times V$.
- $c: A \to \mathbb{R}_+$ be the arc cost function, encoding the ground metric between pixels.

### Kantorovich–Wasserstein Distance

Given two discrete probability measures $\mu$ and $\nu$ on $V$, the **Kantorovich–Wasserstein distance of order 1** with ground cost $c$ is defined as

$$\mathcal{W}_c(\mu, \nu) = \inf_{\pi \in \Pi(\mu,\nu)} \sum_{(x,y) \in V \times V} c(x,y)\, \pi(x,y),$$

where $\Pi(\mu,\nu)$ is the set of all probability measures on $V \times V$ with marginals $\mu$ and $\nu$. When the cost $c$ corresponds to a shortest-path metric on $G$, this is equivalent to an uncapacitated minimum-cost flow problem:

$$\mathcal{W}_c(\mu, \nu) = \min_{x_{ij} \geq 0, \ (i,j)\in A} \sum_{(i,j) \in A} c_{ij}\, x_{ij}$$

$$\text{s.t.} \qquad \sum_{j:\,(i,j)\in A} x_{ij} - \sum_{j:\,(j,i)\in A} x_{ji} = \mu(i) - \nu(i), \qquad \forall\, i \in V.$$

### Wasserstein Barycenter

The **Wasserstein barycenter** of $\mu_1,\ldots,\mu_K$ is the probability measure $z^* \in \mathbb{R}^n_+$ solving

$$z^* = \text{argmin}_{\substack{z \geq 0, \ \sum_{i \in V} z(i)=1}} \sum_{k=1}^{K} \mathcal{W}_c(\mu_k, z).$$

### Joint LP Formulation

The key contribution of the paper is to reformulate the barycenter problem as a **single LP** by writing each Wasserstein distance in its flow form and coupling all $K$ flow problems through the shared unknown barycenter $z$. Introducing flow variables $x^k_{ij} \geq 0$ for each arc $(i,j) \in A$ and each input measure $k$, the barycenter LP reads:

$$\min_{z \geq 0,\; x^k \geq 0} \quad \sum_{k=1}^{K} \sum_{(i,j) \in A} c_{ij}\, x^k_{ij}$$

$$\text{s.t.} \qquad \sum_{j:\,(i,j)\in A} x^k_{ij} - \sum_{j:\,(j,i)\in A} x^k_{ji} = \mu_k(i) - z(i), \qquad \forall\, i \in V,\;\forall\, k = 1,\ldots,K \tag{flow conservation}$$

$$\sum_{i \in V} z(i) = 1 \tag{normalisation}$$

$$x^k_{ij} \geq 0, \quad z(i) \geq 0.$$

The optimal $z^*$ is then the Wasserstein barycenter.

---

## Graph Structures and Ground Metrics

The arc set $A$ and costs $c_{ij}$ determine which ground metric is approximated (or exactly computed). The paper proposes three structured graphs.

### L1 Ground Distance (`BarycenterL1`)

The 4-connected grid graph: arcs connect each pixel $v = (i,j)$ to its horizontal and vertical neighbours only, each with unit cost. The $\ell_1$ distance between adjacent pixels is exactly 1, so the shortest-path distance on this graph equals the $\ell_1$ norm. Formally:

$$A_{\ell_1} = \bigl\{\,((i,j),(i',j')) \;\big|\; |i-i'| + |j-j'| = 1\bigr\}, \qquad c_{ij} = 1.$$

### L∞ Ground Distance (`BarycenterLinf`)

The 8-connected grid graph (king-move graph): arcs connect each pixel to its 8 neighbours (horizontal, vertical, and diagonal), each with unit cost. The shortest-path distance on this graph equals the $\ell_\infty$ norm between pixels:

$$A_{\ell_\infty} = \bigl\{\,((i,j),(i',j')) \;\big|\; \max(|i-i'|, |j-j'|) = 1\bigr\}, \qquad c_{ij} = 1.$$

### L2 Ground Distance (`BarycenterL2`, `BuildGraph`)

For the Euclidean ($\ell_2$) ground distance, an exact sparse graph does not exist. The paper proposes an **approximation** controlled by an integer parameter $L \geq 1$. Define the set of coprime direction vectors

$$\mathcal{C}(L) = \bigl\{\,(v,w) \in \mathbb{Z}^2 \;\big|\; -L \leq v,w \leq L,\; (v,w) \neq (0,0),\; \gcd(|v|,|w|) = 1\bigr\}.$$

Each pixel $(i,j)$ is connected to $(i+v, j+w)$ whenever $(v,w) \in \mathcal{C}(L)$ and the target lies inside the grid, with arc cost $c = \sqrt{v^2+w^2}$:

$$A_{\ell_2}(L) = \bigl\{\,((i,j),(i+v,j+w)) \;\big|\; (v,w)\in\mathcal{C}(L),\; 1 \leq i+v, j+w \leq N\bigr\},$$

$$c_{(i,j),(i+v,j+w)} = \sqrt{v^2 + w^2}.$$

Larger $L$ gives a denser graph and a better approximation of the true $\ell_2$ ground distance.

### Exact Bipartite Formulation (`BarycenterBipartite`)

For comparison purposes, the code also provides an exact solver using a **complete bipartite graph** between two copies of the pixel grid. Every pair of pixels $(u, v) \in V \times V$ is connected by a directed arc with cost equal to the chosen ground metric (L1, L2, or L∞). This yields the classical transportation problem formulation and is exact for any ground metric, at the cost of $n^2$ arcs instead of $O(n)$.

---

## Implementation Details

The code is implemented in Python using:

- **[Gurobi](http://www.gurobi.com/)** (`gurobipy`) — LP solver (barrier method, no crossover).
- **[NetworkX](https://networkx.github.io/)** — Graph construction utilities.
- **[NumPy](https://numpy.org/)** — Input data handling and image normalisation.

Each input image is loaded from a CSV file and normalised so that its pixel values sum to 1 (making it a valid discrete probability measure). The LP is solved with the interior-point (barrier) method (`GRB.Param.Method = 2`) for stability.

### Entry Points

| Function                         | Ground metric      | Graph type                                 |
| -------------------------------- | ------------------ | ------------------------------------------ |
| `BarycenterL1(images)`           | $\ell_1$           | 4-connected grid                           |
| `BarycenterLinf(images)`         | $\ell_\infty$      | 8-connected grid                           |
| `BarycenterL2(images, G)`        | $\ell_2$ (approx.) | Coprime-direction graph with parameter $L$ |
| `BarycenterBipartite(images, G)` | Any                | Complete bipartite graph                   |

The `TestPaper*` functions reproduce the numerical experiments described in the paper for the MNIST handwritten digits dataset.

---

## Requirements

```
gurobipy   (requires a valid Gurobi licence)
networkx
numpy
matplotlib
```
