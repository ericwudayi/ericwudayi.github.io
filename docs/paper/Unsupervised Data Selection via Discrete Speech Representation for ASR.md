---
tags: audio, language-model, self-supervised learning 
---
## Unsupervised Data Selection via Discrete Speech Representation for ASR
[arxiv](https://arxiv.org/pdf/2204.01981.pdf)

**Goal**: 
Improve model performance on specific target domain $T$.
For example:
- ASR in very noisy env.
  
- Speech recognition and ASR in small meeting room.

**Condition**:
- Maybe we don't have many data on target domain $T$, but you have another large "unlabeled" dataset $G$. However, this dataset may have too many data that you don't care about.
  
- Maybe you have a general model $M$, and you need to fine-tune this model to fit target domain $T$.
  
**Problem**:

- How to select representative data in $G$ to improve model performance on $T$ domain ?

**BreakingThrough**:
- Use #language-model to sample a subset $g$ from $G$, that $P(g) \sim P(T)$. 
  
- Train two #language-model $P_G$ and $P_T$ on dataset $G$ and $T$, respectively. For sample $g \in G$, it is selected if $\frac{log P_T( g )  - log(P_G(g))}{len(g)}$ is high.

**Common Misunderstanding**:
- Select $g \in G$ that $P_T(g)$ is high. 
 
    $P(g \in G)$ is not 0 or 1, and it is a probability. That means for even for specific $g \in G$, some of $g$ are in $T$ and some are not.

**Results**

![](../../attachments/2022-04-11-21-26-03.png)