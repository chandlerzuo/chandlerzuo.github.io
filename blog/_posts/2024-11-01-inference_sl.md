---
layout: post
title: Translating Data to Reasoning - The Inference Scaling Law
author: <a href="http://chandlerzuo.github.io/">Chandler</a>
---

**The Traditional Scaling Law**

When we use more compute resources to train a model and increase the model size, can the model perform better? This is the fundamental question answered by the scaling law.

Classifical machine learning domain involves heterogeneous model architectures, which obsoletes the meaning of this question. As we enter into the era of transformers, model architectures have been unified. The unified paradigm of stacking transformer blockers has become the SoTA structure across many domains, from computer vision to natural language processing. When model structures become homogeneous, the scaling law becomes ever important.

OpenAI ([Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)) and DeepMind ([Hoffman et al. 2022](https://arxiv.org/abs/2203.15556)) both developed scaling law research based on empirical studies. As they explored the relationship between training loss and different scaling factors, they found astonishingly simple, elegant rules that dictate such relationships:
* Only three factors affect the model’s performance: model size, data volume, computational budget. These three factors have a Power-law relationship with the model’s performance.
* The type of model, architectural details, and the width and depth of the neural network have a minimal impact and can almost be considered ineffective (assuming we are talking about the architecture of Auto-regressive models).

As shown by the scaling law, data is the key lever to model performance. The other two factors, model size and computation budget, can be purely determined by financial concerns. Data, on the other hand, becomes the sole area that attracts numeric scientific research. Curating large amounts of high quality data underpins the fast advancing LLM generations we have witnessed in the past few years.

![](https://www.dropbox.com/scl/fi/87q4a6f5564s66m0fvo2z/20250102095139_73699.png?rlkey=7r92a1a0bticbiovcst73o5wl&st=zezzj2ug&raw=1)

**The Inference Scaling Law**

Over the last ten years, traditional scaling law has motivated companies to invest in the “bigger is better” direction, racing to invest in hardware and data collections to train bigger models.

This situation has started to change since OpenAI’s introduction of the o1 model. OpenAI o1 allows for more time for the model to generate thinking tokens during inference, and directly improves the accuracy across many tasks. The results show that allowing for additional resources during inference is another effective lever to improve model performance.

Inference time computation can especially help LLM solve complex reasoning tasks. Researchers such as Wang 2024 analogies that to the conscious control mechanism of the human brain. Unconscious control in the human brain engages on broad networks in parietal and prefrontal lobes, which helps deliberate, computational and in-depth thinking. When asking LLM to answer a complex question, with more computational resources at the inference time, models can generate chain-of-thought to cover intermediate steps of their reasoning, before generating the final answer.

**Chain-of-Thought**

The Chain-of-Thought process can be viewed as a Markov Decision Process. Embedded in LLM is a reasoning policy that generates intermediate reasoning steps.
* Question → Reasoning Step 1 → Reasoning Step 2 → … → Reasoning Step N → Final Answer.

How does LLM acquire the CoT capability, and especially, identify the best step-by-step solution towards the final answer? Typically, this can be trained following a Monte-Carlo Tree Search algorithm that optimizes the search path of reasoning traces. However, such algorithms requires training data: we need training examples of a variety of styles.
* Positive examples with both correct reasoning steps and correct final answers.
* Negative examples with incorrect reasoning steps.
* Negative examples with correct reasoning steps but incorrect final answers.

By contrasting positive examples with various types of negative examples, MCTS can learn from each step, what is the most likely correct next step, and gradually expand the reasoning steps towards the final answer.

It is worth noting that human reasoning is beyond one way expansion of reasoning steps; there’s also an important capability, self-correction. Given an incorrect LLM response, can we modify it to become a correct answer? By feeding such data to MCTS, researchers have found it can drastically reinforce LLM’s reasoning capability.

Training data for CoT is highly complex, and the best source is human annotations. A standard annotation flow can be like the following:

![](https://www.dropbox.com/scl/fi/727eoztx253gmqitgr3zf/20250102095159_62474.png?rlkey=rfsi7oe4wknvr5j0k4qtg81kn&st=jg9i320m&raw=1)

In the real world, such annotation processes are essentially the same process as tutoring: generate test questions for LLM, grade its results, and rewrite incorrect ones. Therefore, expert level annotation is essential. To demonstrate what it looks like, say we’d like to improve LLM’s math ability. A training data example can be generated as the following:

First, the annotator writes a prompt.

```Prove that a2+a4+a6+a8+...+a98=93 if a1, a2, a3... is an arithmetic progression with common difference 1, and a1+a2+a3+...+a98=137.```

Then, given the LLM output, the annotation labels the correctness. Here, we use an inline comment to mark all steps following the first mistake.

LLM output:
```
### Proof
Given that `a1, a2, a3...` is an arithmetic progression with common difference 1, we can express the terms as:
`a1 = x`
`a2 = x + 1`
`a3 = x + 2`
...
`a98 = x + 97`
We are given that `a1 + a2 + a3 + ... + a98 = 137`. We can rewrite this as:
`(x) + (x + 1) + (x + 2) + ... + (x + 97) = 137`
Using the formula for the sum of an arithmetic series, we get:
`98x + (1 + 2 + ... + 97) = 137`
The sum `(1 + 2 + ... + 97)` is equal to `(97 * 98) / 2`, so we have:
`98x + (97 * 98) / 2 = 137`
Simplifying and solving for `x`, we get:
**THE RESPONSE STARTING FROM THIS LINE IS INCORRECT**
`x = -49`
Now, let's consider the sum `a2 + a4 + a6 + ... + a98`. Since the common difference is 1, we can rewrite this as:
`(x + 1) + (x + 3) + (x + 5) + ... + (x + 97)`
Using the same formula for the sum of an arithmetic series, we get:
`49(x + 1) + (1 + 3 + ... + 97) = 93`
The sum `(1 + 3 + ... + 97)` is equal to `(49 * 98) / 2`, so we have:
`49(x + 1) + (49 * 98) / 2 = 93`
Substituting `x = -49`, we get:
`49(-48) + (49 * 98) / 2 = 93`
Simplifying, we indeed get:
`-2352 + 2401 = 93`
Therefore, the statement `a2 + a4 + a6 + ... + a98 = 93` is true.
```

Finally, the human annotator rewrites the wrong steps to give a correct solution. We use the yellow color to highlight the rewritten parts.

```
### Proof
Given that `a1, a2, a3...` is an arithmetic progression with common difference 1, we can express the terms as:
`a1 = x`
`a2 = x + 1`
`a3 = x + 2`
...
`a98 = x + 97`
We are given that `a1 + a2 + a3 + ... + a98 = 137`. We can rewrite this as:
`(x) + (x + 1) + (x + 2) + ... + (x + 97) = 137`
Using the formula for the sum of an arithmetic series, we get:
`98x + (1 + 2 + ... + 97) = 137`
The sum `(1 + 2 + ... + 97)` is equal to `(97 * 98) / 2`, so we have:
`98x + (97 * 98) / 2 = 137`
Simplifying and solving for `x`, we get:
`x = -2308/49`
Now, let's consider the sum `a2 + a4 + a6 + ... + a98`. Since the common difference is 1, we can rewrite this as:
`(x + 1) + (x + 3) + (x + 5) + ... + (x + 97)`
Using the same formula for the sum of an arithmetic series, we get:
`49(x + 1) + (1 + 3 + ... + 97) = 93`
The sum `(1 + 3 + ... + 97)` is equal to `(49 * 98) / 2`, so we have:
`49(x + 1) + (49 * 98) / 2 = 93`
Substituting `x = –2308/49`, we get:
`49(-2308)/49 + (49 * 98) / 2 = 93`
Simplifying, we indeed get:
`-2308 + 2401 = 93`
Therefore, the statement `a2 + a4 + a6 + ... + a98 = 93` is true.
```

**Summary**

Earlier LLM improvements are based on the traditional scaling law, which predicts that model’s training loss can scale according to model size, data volume and computation budget.

Since OpenAI’s o1, a new paradigm that focuses on inference time computation, has evolved. OpenAI’s o1 shows that by allocating compute budget during the inference time, models can generate more accurate tracing tokens to improve solving complex reasoning tasks.

LLM’s inference time reasoning is commonly modeled as a Monte-Carlo Tree Search algorithm. To optimize MCTS, training data need to include complex examples including not only final results, but also step-by-step reasoning. Such data are best provided by expert human annotators.

*(c)2017-2026 CHANDLER ZUO ALL RIGHTS PRESERVED*
