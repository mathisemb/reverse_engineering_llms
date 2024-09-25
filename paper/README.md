# Experiments for a possible paper on reverse-engineering LLMs

## Universal Attacks set up
As the current best discrete prompt optimization algorithm is Greedy Coordinate Gradient introduced in the paper [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043), we choosed to use the same set up for our experiments.

They designed a benchmark, *AdvBench*, based on two different settings.
- **Harmful Strings**: a dataset of 500 strings that an aligned LLM should not generate. The adversary's objective is to find an input that makes the LLM generates the exact string.
- **Harmful Behaviors**: a dataset of 500 (goal, target) pairs. The goal is a string asking for harmful content and the target is the string "Sure: Here is" followed by the corresponding harmul content. The adversary's objective is to find a prefix such that prefix+goal makes the LLM generates a reasonable attempt at executing the behavior.

There exist 2 types of **Harmful Behaviors** attack (**Harmful Strings** are necessarily individual).
- **individual Harmful Behavior**: we train one prefix specific to one harmful behavior.
- **multiple Harmful Behavior**: we train one prefix that enables multiple harmful behaviors (just by changing the goal).

Universal Attacks also focus on transfering attacks to multiple models but we won't extend our experiments to this setting.

## Todo list
### Adversarial setting
- [ ] With the 3 settings in this order: **individual Harmful Behavior** -> **multiple Harmful Behavior** -> **Harmful Stirngs**
  - [ ] Naive projections after continuous optimization (negative results)
    - [ ] Cosine
    - [ ] Dot product
    - [ ] L2
  - [ ] GCG from Universal Attacks
- [ ] If our method does not perform better than GCG
  - [ ] Add AutoPrompt to the experiments

### "Prompt-engineering" setting
We now want to optimize the prefix to perform various NLP tasks such as reasoning (and maybe find the one to which our method is the most adapted).
- [ ] Compare GCG and our method
- [ ] Intuitions on the likelihood of the adversarial prefix and generalization
  - [ ] If I get the code from FluentPrompt, experiments on it
