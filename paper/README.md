# Additionnal experiments for a possible paper on reverse-engineering LLMs

We initially aimed to publish a paper but had some surprises with the experiments. This is detailed in ```conclusion.pdf```.

## ```paper``` folder overview
- The ```gcg``` folder contains the [Universal and Transferable Adversarial Attacks on Aligned Language Models code](https://github.com/llm-attacks/llm-attacks) and some code of mine, especially the code of our divergence loss method. You can find a specific README in the folder.
- The ```fixed_point_interpretation.py``` file runs prompt tuning and interprete the resulting adversarial embedding through a sort of fixed point process by asking the LLM *"Modify " + discrete_adv + " so it has the same meaning as " + continuous_adv + ". Sure here is a modification of" + discrete_adv + ":"*. You need to run it from the ```paper``` folder:
    ```bash
    python3 fixed_point_interpretation.py
    ```
- The ```interesting_interpretations.txt``` file contains some interesting examples of the LLM interpreting the meaning of optimized adversarial prompts. In some of them we can almost recognize patterns of the human written adversarial prompts.
- The ```interpretation.py``` file computes the success rate of prompt tuning followed by the LLM interpretation. You need to run it from the ```paper``` folder:
    ```bash
    python3 interpretation.py
    ```
- The ```iterative_interpretation.py``` file computes the success rate of prompt tuning while projecting with the LLM interpretation each $k$ steps. You need to run it from the ```paper``` folder:
    ```bash
    python3 iterative_interpretation.py
    ```
- The ```iterative_naive_projection copy.py``` file computes the success rate of prompt tuning while projecting with a naive projection (dot product, cosine, L2) at each step. You need to run it from the ```paper``` folder:
    ```bash
    python3 iterative_naive_projection.py
    ```
- The ```naive_projection.py``` file compute the success rates of prompt tuning followed by the naive projections: dot product, cosine and L2. You need to run it from the ```paper``` folder:
    ```bash
    python3 naive_projection.py
    ```
- The ```peft_bug.py``` file exhibits a bug I have with the prompt tuning peft model. For an *input+target* sequence, I do not obtain the same logits for *input* doing *model(input)* or *model(input+target)* though the ouputs should not depend on the right context... You can run it like this:
    ```bash
    python3 peft_bug.py
    ```
- The ```random_attack.py``` file computes the success rate of attacking by simply concatenating random tokens. Useful to check if the interpretation attack really works or not. You need to run it from the ```paper``` folder:
    ```bash
    python3 random_attack.py
    ```
- The ```target.py``` file computes the success rate of asking for harmful content without any adversarial prompt. Rather by providing the target in input (to push the LLM generating the rest of the text) or only by giving the goal. You need to run it from the ```paper``` folder:
    ```bash
    python3 target.py
    ```
- The ```utils.py``` file contains losses, training functions, success checking functions and other utility functions.
- The ```variations_bug.py``` file exhibits a small variation in the *goal* logits of the initial model (not peft) depending on whether we use *model(goal)* or *model(goal+target)*. You can run it like this:
    ```bash
    python3 variations_bug.py
    ```

## Universal Attacks set up
As the current best discrete prompt optimization algorithm is Greedy Coordinate Gradient introduced in the paper [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043), we choosed to use the same set up for our experiments.

They designed a benchmark, *AdvBench*, based on two different settings.
- **Harmful Strings**: a dataset of 500 strings that an aligned LLM should not generate. The adversary's objective is to find an input that makes the LLM generates the exact string.
- **Harmful Behaviors**: a dataset of 500 (goal, target) pairs. The goal is a string asking for harmful content and the target is the string "Sure: Here is" followed by the corresponding harmul content. The adversary's objective is to find a prefix such that prefix+goal makes the LLM generates a reasonable attempt at executing the behavior.

You can find those two datasets in the ```paper/data``` folder.

There exist 2 types of **Harmful Behaviors** attack (**Harmful Strings** are necessarily individual).
- **individual Harmful Behavior**: we train one prefix specific to one harmful behavior.
- **multiple Harmful Behavior**: we train one prefix that enables multiple harmful behaviors (just by changing the goal).

Universal Attacks also focus on transfering attacks to multiple models but we won't extend our experiments to this setting.
