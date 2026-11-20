---
layout: post
title: Proof Positive - Maths Has Fallen to the Machines. What Falls Next?
author: <a href="http://chandlerzuo.github.io/">Chandler</a>
topic: ml
---

*The answer has less to do with equations than with referees*

In 1946 Paul Erdős asked how many pairs of points in a plane can sit exactly one unit apart, and he guessed the answer. For 80 years nobody proved him right or wrong. In May 2026 a model built by OpenAI produced a counterexample. Almost at once nine distinguished mathematicians, among them Noga Alon, Timothy Gowers and Melanie Matchett Wood, had [checked it, tidied it up and written down their reflections](https://arxiv.org/abs/2605.20695).

The unit-distance problem was not a one-off. In 2024 DeepMind's systems scored [a silver medal](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) at the International Mathematical Olympiad. A year later they scored [gold](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/). Hobbyists armed with chatbots began knocking off [Erdős problems](https://www.quantamagazine.org/why-the-legendary-erdos-problems-are-falling-to-ai-20260803/) that had defeated professionals for decades. In October 2026 OpenAI dumped [722 manuscripts](https://github.com/openai/math) onto the internet, claiming progress on hundreds of open questions, including a few that whole careers have been built around. Three were withdrawn within a day. Gil Kalai, a mathematician at the Hebrew University of Jerusalem, [noted drily](https://gilkalai.wordpress.com/2026/10/07/updates-sharing-ai-progress-on-mathematics-amazing-and-my-lecture-plans/) that the lot would need to be verified by humans in the months to come. Mathematics, once the standard example of what computers could not do, has become the first intellectual field they are visibly conquering.

So which falls next? The popular answer is whatever can be written in equations. Maths is the most formal discipline, the argument runs, so the most mathematical sciences will follow it: theoretical physics first, then economics, then anything with enough Greek letters. That answer is wrong, and the reason it is wrong says a lot about where AI will and will not succeed.

**The referee, not the rulebook**

Maths did not fall first because it is the hardest thing people do. It fell first because it has the best referee. A proof written in Lean, a programming language for mathematics, is checked by a small piece of software that accepts or rejects it in seconds, at almost no cost, and cannot be talked into anything. A learning system can therefore be marked millions of times a day, without error. That is how DeepMind trained AlphaProof, and broadly how the "reasoning" models of the past two years were taught to think.

The idea is older than the chatbots. In 2001 Rich Sutton, a pioneer of reinforcement learning, [put it plainly](http://incompleteideas.net/IncIdeas/KeytoAI.html): "An AI system can create and maintain knowledge only to the extent that it can verify that knowledge itself." Jason Wei, an AI researcher, [compressed it](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law) into a rule in 2025: the ease of training AI on a task is proportional to how verifiable the task is.

Equations mattered in maths only because they produced a perfect referee. Elsewhere the two come apart. A model of the economy can be as rigorous as a theorem, but the economy, not a proof checker, decides whether it is right.

**Six questions for any field**

To forecast which field falls next, ask six questions about its referee:

- *How quickly, and how cheaply, does a verdict arrive?* Seconds for a proof, months for a crystal structure, years for a drug, decades for a monetary policy.
- *Is the verdict the real thing, or a stand-in?* A simulation is not an experiment, and a p-value is not a true effect. A system that optimises hard against a stand-in learns to satisfy the stand-in.
- *Is there a back catalogue of past verdicts to learn from?*
- *Can the machine run the test itself,* or must a robot, a hospital or a parliament do it?
- *Does the subject sit still* when it is studied, or does it change because someone acted on a prediction?
- *Does anyone agree on what counts as right?*

Notice what is missing from the list: how mathematical the field is. Rank the disciplines by these questions and a clear pattern emerges.

| Field | Who gives the verdict | Prognosis |
|---|---|---|
| Maths, software | A proof checker, a test suite | Falling now |
| Weather, protein structure | Tomorrow's weather; decades of lab results | Already fallen |
| Chemistry, materials | Synthesis and measurement | Within years, at the pace of robot labs |
| Medicine, experimental physics, climate | Clinical trials, colliders, time | Slowly |
| Finance | The market, which then moves | Never settled |
| Macroeconomics, social science | History, which runs once | Not in the way maths has |
| Law, ethics, art | People, who disagree | Machines join in; they do not settle |

**The next to go**

Software is maths's closest cousin. Compilers and tests are fast, nearly free referees, which is why AI coding has improved faster than almost anything else. The referee is imperfect, though: tests check only what someone thought to test. Donald Knuth, a computer scientist, put the problem best in [a memo from 1977](https://www-cs-faculty.stanford.edu/~knuth/faq.html): "Beware of bugs in the above code; I have only proved it correct, not tried it."

Weather forecasting has already gone. It gets a fresh, objective verdict every day and has decades of records to learn from, and DeepMind's [GraphCast](https://www.science.org/doi/10.1126/science.adi2336) beat the best conventional system on most measures. Protein structure fell to [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2), which won Demis Hassabis and John Jumper a share of the 2024 Nobel prize in chemistry. The machine was clever, but the crystallographers had done the hard part. They had spent half a century building a library of some 170,000 solved structures, plus a blind contest, CASP, in which predictions have been scored against unpublished answers since 1994. AlphaFold was, in effect, trained on the accumulated output of a very slow referee.

Chemistry and materials science are next, and they show what happens when the referee is only a stand-in. In 2023 Google DeepMind's [GNoME](https://www.nature.com/articles/s41586-023-06735-9) predicted that 2.2m new crystals would be stable. The stability was calculated, not measured, and two eminent materials scientists, Anthony Cheetham and Ram Seshadri, [argued](https://doi.org/10.1021/acs.chemmater.4c00643) that many of the compounds were neither new nor useful. A robotic laboratory that claimed to have [made dozens of the targets](https://www.nature.com/articles/s41586-023-06734-w) in 17 days drew [similar scepticism](https://doi.org/10.1103/PRXEnergy.3.011002). The machines are very good at passing the test. The remedy is to make the test real, with robots that synthesise and measure. Progress will then be paced by the robots, not the models.

**The slow lane**

Medicine has an excellent referee, the randomised trial, but a ruinously slow and expensive one. The data show exactly where the machines' advantage stops. Molecules discovered by AI pass early safety trials [80-90% of the time](https://doi.org/10.1016/j.drudis.2024.104009), well above the industry's historical rate, because drug-likeness can be learned from a large back catalogue. In Phase II, where the question becomes whether the drug actually works in patients, their success rate is around 40%, no better than anyone else's. Rentosertib, a drug for lung fibrosis whose target and molecule were both found with generative AI, completed a [71-patient trial](https://doi.org/10.1038/s41591-025-03743-2) in 2025. The bigger trials will take years. Biology answers at the speed of patients, however fast the algorithms are.

Experimental physics, climate science and ecology face the same constraint. AI will speed up the analysis of every observation, but the observations come from particle colliders, satellites and field seasons. A 50-year climate projection gets its verdict in 50 years. These fields will fall piecemeal: the parts with quick, local answers first, the big questions last.

**The ones that won't**

Finance is the instructive exception. It has fast, accurate verdicts (prices), oceans of data, and a machine can place its own trades. Yet it can never be conquered, because every edge that is found and exploited changes the market. R. David McLean and Jeffrey Pontiff [found](https://doi.org/10.1111/jofi.12365) that published stock-market anomalies earn 26% less out of sample and 58% less once they are published. In markets AI does not end the game. It raises the standard of every player.

Macroeconomics is the sharpest rebuke to the idea that equations are destiny. Few disciplines are more mathematical, and few have a worse referee. History runs once, verdicts take decades, and the economy reacts to the models made of it, as Robert Lucas [pointed out](https://doi.org/10.1016/S0167-2231%2876%2980003-6) in 1976. Paul Romer, a Nobel laureate in economics, coined the word ["mathiness"](https://doi.org/10.1257/aer.p20151066) for rigorous equations loosely tied to anything measurable. A cleverer model does not fix that.

The other social sciences run into a ceiling on predictability. In the [Fragile Families Challenge](https://doi.org/10.1073/pnas.1915006117), 160 teams armed with machine learning and an unusually rich dataset tried to predict how children's lives would turn out. The best did only slightly better than a simple baseline. The referee is shaky too. When a [large project](https://doi.org/10.1126/science.aac4716) re-ran 100 psychology studies, 97% of the originals had reported significant results but only 36% of the repeats did.

Then there are law, ethics, policy and art, where there is no correct answer to find, only the judgment of people who disagree with each other. Machines will become very good at producing what judges, voters and audiences approve of. That is a different achievement, and it has a familiar failure mode: language models trained on human approval learn [sycophancy](https://arxiv.org/abs/2310.13548), telling people what they want to hear. Even maths has a corner like this. William Thurston, a Fields medallist, [argued](https://arxiv.org/abs/math/9404236) that the point of mathematics is human understanding. A proof checker can say whether a theorem is true, but not whether it was worth proving.

**Build the referee**

The fields AI conquers have the same profile: a closed loop with a quick, cheap, faithful and stable referee, and agreement on what winning means. The holdouts fail in one of four ways. The verdict is slow (medicine). It is a stand-in (simulated materials, p-values, applause). The subject answers back (markets, economies). Or nobody agrees on the verdict at all (law, art).

For businesses, and for the governments and foundations that fund research, the lesson follows directly. The scarce resource is no longer cleverness, which is getting cheap. It is verification. Lean did for maths what the Protein Data Bank and CASP did for biology. Other fields need their equivalents: robot labs that close the loop, open data, blind benchmarks, routine replication, prediction contests scored against outcomes not yet known. A field that builds its referee will be transformed. One that does not will mainly get a flood of plausible-looking papers, which is what the 722 manuscripts amount to until someone checks them.

There is a philosophical bonus. Philosophers have long argued about what separates science from pseudoscience, and for a while the favoured answer was formal rigour. Astrology can be formalised perfectly well, and it [failed a blind test](https://doi.org/10.1038/318419a0) in 1985. A notorious study of precognition passed standard statistical tests in [eight of nine experiments](https://doi.org/10.1037/a0021524). Darwin's "Origin of Species" contains almost no mathematics. The better answer, running from Karl Popper's falsifiability to Deborah Mayo's ["severe testing"](https://doi.org/10.1017/9781107286184), is that a claim is scientific if it submits to a check it does not control and changes when it fails. Pseudoscience, in Sven Ove Hansson's [definition](https://plato.stanford.edu/entries/pseudo-science/), imitates the trappings of science while avoiding the check. That is also, it turns out, the best predictor of where AI will succeed. The machines are learning what scientists were supposed to know all along: being rigorous is not the same as being right, and it is the referee who decides.

*(c)2017-2026 CHANDLER ZUO ALL RIGHTS PRESERVED*
