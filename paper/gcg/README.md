# LLM Attacks

The ```paper/gcg/llm-attacks``` contains the code of the paper ["Universal and Transferable Adversarial Attacks on Aligned Language Models"](https://arxiv.org/abs/2307.15043).

It implements the Greedy Coordinate Gradient method.

You need to install the requirements by running ```pip install -r requirements.txt``` in the ```paper/gcg/llm-attacks``` folder.

## What I added to this ```llm-attacks``` folder?
- The ```paper/gcg/llm-attacks/gcg.py``` file which is the python script used to compute the success rate of the individual harmful behaviors experiment. You need to run it from the ```paper``` folder.
- The ```paper/gcg/llm-attacks/my_experiments``` folder which contains python scripts for some the experiments mentionned in my internship report.
  - The ```one_example``` folder contains the scripts to run autoprompt, gcg and our method on a single example.
  - The ```attraction_test.py``` file aims to check if we can noise an input embedding such that the LLM cannot answer it corectly but can after optimizing the input with the loss attracting the embeddings toward the embedding matrix.
  - The ```autoprompt.py``` file defines a function that runs autoprompt on one example (adapted from the gcg code).
  - The ```entropy_test.py``` file checks two implementation of the entropy.
  - The ```gcg.py``` file defines a function that runs gcg on one example.
  - The ```launch_nbsteps_comparison.py``` file runs the prompt_tuning, our_method, gcg and autoprompt methods on a dataset. Each method is ran until it succeeds and we save the number of steps it took to succedd. We can then average on the dataset.
  - The ```launch_success_comparison.py``` file runs the prompt_tuning, our_method, gcg and autoprompt methods on a dataset. Each method is ran for a fixed number of steps. For each example of the dataset we save if the moethod succeeded or not and then we can get an average success rate on the dataset.
  - The ```my_attack.py``` file contains some losses and the function that computes gradient of the loss with respect to the prompt.
  - The ```one_prompt_tuning.py``` file corresponds to manual prompt tuning with the gradient computation function from the llm-attacks code.
  - The ```our_method.py``` file defines a function that runs our method on one example.
  - The ```prompt_tuning.py``` file defines a function that runs manual prompt tuning on one example.
  - The ```run_one.py``` file runs our method on one example.
  - The ```utils.py``` file contains some utility functions such as checking if a prompt achieves a successful attack or measuring distance of embeddings to the embedding matrix.
