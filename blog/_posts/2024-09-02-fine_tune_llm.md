---
layout: post
title:  Fine Tuning LLM - A Primer
author: <a href="http://chandlerzuo.github.io/">Chandler</a>
---

**A Brief History of NLP and LLM**

The past few years have witnessed a rapid development of LLM. To understand the context of how LLM has achieved its status today, it is worth taking a brief look at the history of Natural Language Processing predating today’s LLM.

The development of Machine Learning can be roughly divided into five stages.
* Before 1980, NLP relied on a rule-based system on small-scale data processing.
* In the 1980s, initial Machine Learning techniques were developed, which allowed for learning parameterized models for medium sized data.
* Between 1990 and 2014, an important sub-field of Machine Learning, Deep Learning, came into place. Deep Learning used Neural Network models that mimicked the structure of human brains. A variety of NN models were introduced during this era, from CNN, RNN to GAN.
* In 2017, a major breakthrough happened, when Transformers was introduced. The transformer model introduced a generic attention-based structure, intended to mimic the human brain’s ability to distinguish and aggregate the most important features when processing complex real world information. Transformers turned out to be a generic, powerful model that greatly excelled over earlier NN models across many application tasks, including natural language processing, computer vision, and personalization.
* Since 2017, the NLP field has focused on scaling transformer models to solve more complex tasks. As researchers scaled transformer models from tens of millions to hundreds of billions of parameters, they noted that the model acquired emerging AI capabilities such as acquiring common sense knowledge, being able to chat with people and solving complex reasoning tasks. Thus came the age of LLM.

Today, researchers across the world are continuing to improve LLM to better hone their AI capabilities. In a nutshell, such capabilities can be categorized into two areas:
* Communicate like human beings. LLMs can understand natural languages and can generate responses.
* Learn like human beings. LLM can transfer knowledge across different domains so that they can acquire complex capabilities like humans, for example, solving Olympic level math problems.

**The Training Paradigm in LLM**

The model training framework of LLMs is different from traditional machine learning.
Traditional machine learning developed tailored models for different tasks. For example, there were different models for machine translation, sentiment analysis and question-answers. Each of these models has their specific architecture and is trained independently, using a large amount of training data.

In LLM, model training is broken down into two stages: a pre-train stage and a fine-tune stage.
* The pre-train model is universal across all tasks. It is trained using a vast amount of data across all domains. As a result, the pre-trained model is preoccupied with general knowledge about the world.
* The fine-tune stage is still task-specific. For each task, a small amount of data is fed into the pre-train model to hone a specific capability, for example, solving math problems. Because the pre-trained model is already a powerful assembly of human knowledge, the fine-tune stage needs a small amount but super high quality data.

**How Data Affect LLM Training**

Due to the different scale and purpose of pre-train and fine-tune, the data requirement is very different from the two stages.

During pre-train, data volume, data quality and diversity are the most important requirements. Pre-training LLM today involves training on almost all available human generated text data. Careful data cleaning is needed to filter out noisy, uninformative texts. The data also need to balance across different domains to ensure LLM can effectively transfer the knowledge, and acquire the potential to be fine tuned for all downstream tasks.

During fine-tune, quality is the single most important requirement. Pre-trained LLM already has acquired general knowledge about the world, therefore, common data we encounter in our daily life, such as web forum discussions, can no longer benefit LLM in specific areas. Rather, we need expert-level data in a specific domain. For example, using PhD level text in STEM domains to enhance LLM to solve advanced problems in these areas.

The specific quality requirement during the fine-tuning stage asks for strong demand for expert-level data annotations. This is because there’s a scarcity of expert-level data on the web. To bring expert-level intelligence to LLM in specific domains, there’s no alternative rather than leveraging human experts to iteratively interact with LLMs, feed complex data and provide feedback to LLM responses, to continue enhancing their domain specific intelligence.

**Fine Tuning Algorithms**

Typically, LLM fine tuning involves two steps.
* Supervised Fine Tuning (SFT). SFT focuses on teaching LLM what type of responses do users expect. Training data for SFT involves input output examples. The inputs are called “prompts”, which LLM receives, and the outputs are called “responses” which are the expected response from LLM. Often, the input-output pairs are generated by human annotation to ensure their high quality. 
* Reinforcement Learning Human Feedback (RLHF). RLHF focuses on teaching LLM human preferences. Compared to SFT, RLHF in addition teaches LLM what users do NOT expect. In order to do so, RLHF utilizes the RL algorithm. It iteratively prompts the LLM to generate various responses, and uses a reward model to score them according to human preferences. Training data at this stage involves preferential labeling on LLM responses. By training LLM on annotated data that expresses human preferences, RLHF encourages LLM to better align its output on human preferences.
* In recent research such as Llama, a new algorithm, Direct Preferential Optimization (DPO), has been developed. DPO unifies SFT and RLHF into one single step and avoids training instability issues for RL algorithms.

**A Fine Tuning Example**

Let’s take a look at a real world fine tuning example. Suppose we want to develop a LLM that can generate coding solutions given a coding problem. How to do so?

* Choose a pre-trained LLM. In this case, if we expect English to be the main use case, an open source pre-trained LLM that focuses on English is a reasonable choice, for example, Llama. If we expect the model to have English-Chinese bilingual behaviors, we can choose an English-Chinese bilingual pre-trained LLM such as Qwen2.
* Prepare fine tuning data. For SFT, training data consist of correct question-solution pairs. For example,
```{
“Question”: “Write a function to generate the nth Fibonacci number. The function should have a time complexity of O(log n) and use dynamic programming. Additionally, the function should only use a constant amount of space, without using any arrays or additional data structures.”,
“Answer”: “def multiply_matrices(A, B): a, b, c = A[0][0], A[0][1], A[1][0] d, e, f = B[0][0], B[0][1], B[1][0] return [ [a*d + b*e, a*e + b*f], [c*d + e*d, c*e + e*f] ] def power_matrix(A, n): if n == 0: return [[1, 0], [0, 1]] # Identity matrix result = power_matrix(A, n // 2) result = multiply_matrices(result, result) if n % 2 == 1: result = multiply_matrices(result, A) return result def fibonacci(n): if n == 0: return 0 matrix = [[1, 1], [1, 0]] result = power_matrix(matrix, n - 1) return result[0][0]”
}
```
For RLHF, training data involves multiple answers with labeled preference, i.e.
```
{
“Question”: “Write a function to generate the nth Fibonacci number. The function should have a time complexity of O(log n) and use dynamic programming. Additionally, the function should only use a constant amount of space, without using any arrays or additional data structures.”,
“Chosen Answer”: “def multiply_matrices(A, B): a, b, c = A[0][0], A[0][1], A[1][0] d, e, f = B[0][0], B[0][1], B[1][0] return [ [a*d + b*e, a*e + b*f], [c*d + e*d, c*e + e*f] ] def power_matrix(A, n): if n == 0: return [[1, 0], [0, 1]] # Identity matrix result = power_matrix(A, n // 2) result = multiply_matrices(result, result) if n % 2 == 1: result = multiply_matrices(result, A) return result def fibonacci(n): if n == 0: return 0 matrix = [[1, 1], [1, 0]] result = power_matrix(matrix, n - 1) return result[0][0]”,
“Unchosen Answer”: “def matrix_mult(A, B): return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]], [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]] def matrix_pow(mat, exp): if exp == 1: return mat if exp % 2 == 0: half_pow = matrix_pow(mat, exp // 2) return matrix_mult(half_pow, half_pow) else: return matrix_mult(mat, matrix_pow(mat, exp - 1)) def fibonacci(n): if n == 0: return 0 if n == 1: return 1 base_matrix = [[1, 1], [1, 0]] result_matrix = matrix_pow(base_matrix, n - 1) return result_matrix[0][0]”
}
```
* Training the model and evaluating the model performance.
* Iterations. We can iterate the process of data collection and model training multiple times. In each round, we can focus on improving weaknesses identified in the previously fine tuned model. For example,
  * Is the model weak in specific types of coding questions, for example, dynamic programming? If so, collect more data in this area.
  * Is the model weak in solving hard problems? If so, involve experts to create more challenging coding questions for training data.
  * Is the model weak in specific coding languages, for example, SQL? If so, collect more data in such languages.

**Summary**

Recent development in LLM shows that models intended to solve natural language problems exhibit emerging AI behaviors such as solving common sense reasoning, writing codes, and solving STEM questions. This motivates further research to enhance LLM’s capability in these domains.

Modern training paradigm of LLM involves a pre-train stage and a fine-tune stage. Pre-train is both data and infra hungry, and is typically done by a few companies in the world. Fine-tune, on the other hand, focuses on customizing pre-trained LLM for specific tasks. To improve LLM’s capability in specific domains or for customized applications, fine-tune is the most important stage.

The fine-tune stage is heavily driven by high quality data. To guarantee such data can further enhance pre-trained LLM’s capability in specific tasks, we need domain expert-level data. For example, generate compute-optimal solutions for coding questions.

*(c)2016-2025 CHANDLER ZUO ALL RIGHTS PRESERVED*
