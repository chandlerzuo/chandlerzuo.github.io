---
layout: post
title: Is tokenization the Missing Link between LLM and Recommender Systems?
author: <a href="http://chandlerzuo.github.io/">Chandler</a>
---

**THE ATTEMPT OF GENERATIVE RECOMMENDATION**

LLM has demonstrated very powerful Scaling Law behaviors. Simply by stacking homogeneous transformer blockers and training on a simple next-token prediction task, model performance naturally scales with model complexity. Inspired by the success, in recent years, there are many attempts to apply the similar modeling framework in recommender systems, with the expectation of achieving similar scaling law performance.

The results in the industry, however, is mediocre in my opinion. Although notable research has been around for a few years, such as Meta's [HSTU](https://arxiv.org/abs/2402.17152), Deep Mind's [TIGER](https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf), and Kuai's [OneRec](https://arxiv.org/html/2502.18965v1), the most successful models in production still stay using classical model architectures such as [DIN](https://github.com/guyulongcs/Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising/blob/master/03_Ranking/Sequence-Modeling/2018%20%28Alibaba%29%20%28KDD%29%20%2A%2A%5BDIN%5D%20Deep%20Interest%20Network%20for%20Click-Through%20Rate%20Prediction.pdf). In reality, the transformer-based models often cannot beat the performance of classical solutions without significantly increasing the compute budget.

There are many hypotheses why it is hard to transfer the success of transformer architecture from language tasks to the recommender systems domain. In this article, I focus on one specific hypothesis: **the gap in tokenization**.

**THE GAP IN TOKENIZATION**

The input to LLMs is a sequence of tokens. These tokens are from a dictionary of relatively small sizes, e.g. 129k in DeepSeek-V3, that not only covers different languages, but also can convey concepts of *INFINTE* complexity in the *ENTIRE* world. Such an efficient coding system is due to thousands of years of linguistic evolution. LLM is built on top of highly optimized linguistic systems, and takes it for free.

Such luxury is lost for recommender systems. Items are opaque IDs, users are sparse interaction histories, and features are high-dimensional messes without a natural discrete codebook.

Hence is my view: Tokenization is *the* bottleneck. Without a mature, compositional token system that fuses collaborative filtering signals with semantics, LLM architectures can't match mature deep networks.

What is Compositionality? Compositionality is originally a linguistic concept. It means that the meaning of any complex expression is solely determined by the meanings of its individual parts (language tokens) and the syntactic rules to combine them. For a token system in recommender systems, compositionality means that each user/item can be represented by multiple tokens organized in a certain way, and the meaning of such a multi-token representation is determined solely by the meanings of individual tokens as well as some pre-defined rules to combine them.

Why is compositionality needed for the RecSys token system?

* In recommender systems, the token system needs to be able to describe an item pool of cardinality of O(10M) and a user pool of cardinality of O(1B), which is the typical scale of many industry use cases.
* Without compositionality, the only way to use a O(100K) token set to represent O(10M)-O(1B) entities is to group them together according to some notion of similarity, then use a single token to represent each group. However, such a solution won't be able to preserve fine-grained collaborative filtering patterns, one of the most important "power source" for industry recommender systems.
* With compositionality, we can combine multiple tokens together to describe individual users and items, without losing the granularity of such representations.

Along this line, there are quite a few challenges to design such a tokenization system.

* First, to fuse collaborative filtering signals into this token system is a chicken-and-egg problem. The goal of such a system needs to support a downstream LLM to memorize collaborative filtering signals, yet in order to achieve so, the token system also needs to know those collaborative filtering patterns in advance in order to encode them efficiently.
* Second, the token system can be used to describe complex entity relationships, not just a sequence of user engagements. To take a step back, many existing LLM-style recsys models draws an analogy between the next-token prediction task in LLM to the next-action prediction in user engagements. This approach is very limited, because the meaning behind next-token prediction task is highly versatile, covering many domains, with varying language formats, and conveys complex relationships. In contrary, next-action prediction in user engagement is very limited. Successful LLM-style recsys may need to be pre-trained on other tasks to learn other complex relationships such as user social graph, cross-modality consistency, substitute/complementary relationships between products, to name a few. In order to support such alternative training tasks, the token system needs to support the description of such complex entity relationships.


**WHERE WE ARE**

The existing research in tokenization for recsys can be categorized into the following themes.

* No extra tokenization beyond using item IDs, e.g. [HSTU, 2024](https://arxiv.org/abs/2402.17152) and [Netflix, 2025](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39).
* Tokenizing semantic features, e.g. [TIGER](https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf),
* Fusing with CF-based signals. Early research in [UTGRec, 2018](https://arxiv.org/abs/2504.04405) relies on alignment tasks to align semantic-based token space to CF structures. More recent works directly use CF signals within tokenizer inputs to improve the granularity. [TokenRec, 2024](https://arxiv.org/html/2406.10450v3) develops purely CF-based tokenization, [SETRec, 2025](https://arxiv.org/pdf/2502.10833) combines CF signals with semantic signals, and [ETEGRec, 2025](https://arxiv.org/abs/2409.05546) further jointly learns tokenization with generative recommendation models.
* Augmenting with contextual information. Contextual information such as time, location, device info are critical features for recsys, yet there is limited research on how to embed such information within the tokenization. The most common approach is to inject such features downstream, such as in [Netflix, 2025](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39). One novel approach is to use specialized tokens, such as the action tokens in [HSTU, 2024](https://arxiv.org/abs/2402.17152).
* Compositionality. Perhaps the only compositionality structure explored so far is *set compositionality*, i.e. representing each entity by a set of tokens. This structure is used by all methods that inherits the Residual Quantization methods, such as [TIGER](https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf) and [SETRec, 2025](https://arxiv.org/pdf/2502.10833). In LLM, compositionality comes from versatile syntactic structures, which leads to high expressiveness of complex concepts. Comparatively, the expressiveness of set compositionality structure is still very limited.
* Tokenization for non sequential prediction tasks. Although the majority of efforts in recsys scaling law follows the transformer-based next token prediction task paradigm, some research abandons this paradigm and rather sticks with the conventional model archs that focus on deep feature interactions. Tokenization in this paradigm is more flexible as they can leverage more flexible feature structures beyond sequence. E.g. [TokenMixer, 2025](https://arxiv.org/html/2507.15551v1) directly tokenizes existing recsys features. Such an approach can directly leverage the optimized features in existing recsys that is the outcome of years of feature engineering efforts, which can be viewed as an analogy of how those optimized linguistic structures are naturally leveraged by LLMs.

**WHAT IS NEXT**

While existing tokenization efforts have laid important groundwork, from ID-only baselines to hybrid CF-semantic fusions and basic set compositionality, these approaches still fall short of the expressive, evolved power of linguistic tokens that LLMs exploit so effectively. Bridging this gap requires pushing toward token systems that are truly compositional, generalizable and versatile enough to unlock scaling laws in RecSys. In my opinion, the path forward centers on three intertwined research directions.

*Versatility for tasks beyond sequence modeling.*
The majority of LLM-style recsys relies on the "next engagement = next token" analogy. This analogy is too narrow compared to the diverse objectives behind LLM next-token prediction. A successful LLM-style recsys will likely learn the *world knowledge* about recommendations through mixing multiple tasks to learn rich structural signals: user community graphs, multi-hop reasoning over heterogeneous behavior graphs, multi-modal alignment, and substitute/complement relationships in catalogs. To support such tasks, tokenization will also need to encode not only sequence patterns and semantics, but also more complex graph structure and dynamic contexts in the real world, so that the recsys model can act more like a true *world model* over users, items rather than a glorified sequence model.

*Retain the knowledge from existing feature engineering.*
Years of iterative feature engineering in production RecSys have distilled highly optimized feature representations, from graph-based neighborhoods, multi-granularity user profiles, to cross-feature interactions. Such "unshiny" feature engineering efforts are usually the most impactful levers in production recsys, yet most existing work (with TokenMixer being an exception) has discarded such hard-won knowledge in favor of end-to-end learning from scratch. A more promising approach is to discretize these pre-engineered features into reusable primitives. E.g., quantizing DIN-style interest vectors or graph convolutions into token sets, to allow generative recommendation models to bootstrap from proven signals rather than reinventing suboptimal wheels.

*Achieve both granularity and generalizability through rich compositionality.*
Recsys tokenizers constantly face two failure modes: over-collapsing heterogeneous entities into coarse buckets that compromise fine-grained prediction, and over-fragmenting the space that destroys generalization. The compositionality view reframes this as learning a small number of reusable *primitives* that can be combined to describe infinite numbers of contextualized user-item interactions. Concretely, this may look like multi-scale tokenization that combines broad \& fine-grained concepts and static attributes \& temporal dynamic contexts, with richer compositional rules that set compositionality to express entity relationships.

*(c)2017-2026 CHANDLER ZUO ALL RIGHTS PRESERVED*