---
layout: post
title:  The Role of Data Annotation in LLM
author: <a href="http://chandlerzuo.github.io/">Chandler</a>
---

**Data Annotation**

Data annotation is the labeling, tagging of raw data or complete generation of new data for training machine learning models. The emergence of reasoning LLM presents unique challenges as well as new opportunities for data annotation. Well structured high quality annotated data can direct LLM towards developing accurate, robust reasoning structures that can be leveraged to solve complex tasks.

**Types of Annotations for Reasoning**

*Few-shot Examples for In-Context Learning*

Few-shot Learning earning is a training-free method to boost LLM for downstream tasks. When prompting LLM, few-shot learning employs illustrative demonstrations called few-shot examples. It leverages LLMs’ In-Context Learning capability to identify patterns among these demonstrations to solve problems in a new area.

Few-shot learning works the best if few-shot examples are relevant to the new problem. In a new task domain, user-generated prompts may cover all different topics and with difficulty levels. Therefore, a diverse set of few-shot examples need to be curated in advance, so that when given a specific user-generated prompt, we can search from this diverse set to find the most relevant examples as few-shot examples.

The curation of the few-shot example set can either be entirely manual or assisted by LLMs.
* Manual curation. Experts write a set of hundreds & thousands of examples in the task domain.
* LLM-assisted curation. Methods such as [synthetic prompting](https://arxiv.org/abs/2302.00618) allows for expanding a pre-existing example set through LLM generation. When using LLM-assistent example curation, human experts need to be leveraged to ensure the generated examples are of high quality. This can be done by having human experts labeling the quality of LLM generated examples, which are used to train a second quality model to filter LLM generations.

*Chain-of Thought (CoT)*

CoT is the process of solving complex problems step-by-step. To train LLM for better CoT, training data needs to cover a diverse set of examples with step-by-step solutions to different problems, as well as label the correctness of each step.

Leveraging human experts to create step-by-step solutions from scratch is usually challenging. This is because enumerating all possible CoT paths is prohibitive. Think of a math problem that involves 3 steps, where each step there might be 10 options. The total number of possible CoT paths is 10^3=1000. It is impossible to leverage humans to enumerate all possible solutions for diverse problems. A more practical approach is to prompt LLM to generate CoT paths first, then involve human annotation to label the correctness of each step and/or rewrite the step-by-step solutions. Such processes involve human annotations in two places:
* Construct prompts to elicit different CoTs. Experts write a set of prompts, including CoT examples, that can prompt the LLM to generate different CoT paths for each problem. For example, [Wei et al. 2022](https://arxiv.org/abs/2201.11903) shows diverse few-shot CoT examples can drastically improve LLM reasoning for math problems.
* Label and rewrite the LLM CoT responses. Based on the LLM responses, experts label whether individual steps are correct, and rewrite incorrect examples if possible. These annotations will be used to train reward models for [Process Supervision](https://www.bing.com/search?q=Let%27s+Verify+Step+by+Step&cvid=cd024f0c7f94432d9fcb17960ff3f174&gs_lcrp=EgRlZGdlKgYIABBFGDkyBggAEEUYOTIGCAEQABhAMgYIAhAAGEAyBggDEAAYQDIGCAQQABhAMgYIBRAAGEAyBggGEAAYQDIGCAcQABhAMgYICBAAGEDSAQc1MThqMGo0qAIAsAIB&FORM=ANAB01&PC=U531), a critical component to optimize LLM’s reasoning path for new problems.

*Instruction Tuning*

Instruction Tuning is a general purpose technique for LLM fine tuning, and can be viewed as a simplified version of training CoT. Instruction Tuning focuses on the correctness of LLM’s final answer. Therefore, training data include pairs of prompts and the correct final answers only. The exclusion of intermediate steps brings human curation back to the table. Correspondingly, human annotations can be involved in two areas:
* Manual curation. Curate a diverse set of prompt and final answer pairs.
* Label and rewrite the LLM response. Methods such as [Self-Instruct](https://arxiv.org/abs/2212.10560) can be used to use LLM to generate pairs of prompts and answers. These are not guaranteed to be correct, so humans need to be involved to provide the correctness labels. These labels can be used to train [Outcome Supervision](https://arxiv.org/abs/2311.09724) reward models that can be further leveraged for filtering LLM responses.

*Aligning with Human Preferences*

It is important that LLM-based systems are helpful, honest and harmless when interacting with the public. Such attributes are distinct from their reasoning skills, but are widely recognized by the LLM community. These human-centric attributes can be learned through techniques such as [Reinforcement Learning for Human Feedback](https://huggingface.co/blog/rlhf).

Training data in this step involves comparing different responses from LLM according to helpfulness, honesty and harmlessness. The standard practice is to involve human annotators to compare pairs of LLM responses for the same prompt. Such pairwise preferential data are then used by the Reinforcement Learning algorithm to navigate towards the best responses that align with human values.

**Challenges in Data Annotations**

LLM performance strongly depends on the data quality. To ensure quality data in a process deeply involving human efforts, rigorous processes need to be set to control the bias and randomness of human behavior.

* Access to expert-level annotator pool. LLM reasoning focuses on tackling challenging tasks. Access to expert-level annotators across different subjects, such as people with graduate-level degrees in STEM subjects, is necessary.
* Clear annotation guidelines. Detailed, disambiguous instructions need to be provided to data annotators to ensure data annotations are consistent and not affected by individual biases.
* Quality measures. Quality checks throughout the annotation process are important to detect errors. Active analyses need to be performed to identify common quality issues early, which can be used to refine the annotation guidelines.
* Bias mitigation. Statistical analyses need to be implemented to actively monitor the diversity of the dataset, such as the difficulty distribution of STEM datasets. In addition, the diversity in the annotator pool is crucial. To achieve this, the recruiting process needs to be monitored and adjusted constantly.

**Summary**

In this article, we introduce different types of data annotations involved in developing reasoning LLMs. In retrospect, they can be categorized in the following:
* Directly provide a set of examples for LLM training;
* Develop prompts to elicit diverse response from LLM;
* Label & rewrite LLM responses.

In practice, for advanced reasoning tasks, access to a skillful annotator pool is necessary. Besides, annotation platforms also need to implement robust mechanisms to ensure annotation quality and remove human biases.

*(c)2017-2026 CHANDLER ZUO ALL RIGHTS PRESERVED*
