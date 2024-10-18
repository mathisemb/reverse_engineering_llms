# LLM Attacks

The ```paper/gcg/llm-attacks``` contains the code of the paper ["Universal and Transferable Adversarial Attacks on Aligned Language Models"](https://arxiv.org/abs/2307.15043).

It implements the Greedy Coordinate Gradient method.

You need to install the requirements by running
```bash
pip install -r requirements.txt
```
in the ```paper/gcg/llm-attacks``` folder.

## What I added to this ```llm-attacks``` folder?
- The ```paper/gcg/llm-attacks/gcg.py``` file which is the python script used to compute the success rate of the individual harmful behaviors experiment. You need to run it from the ```paper``` folder:
  ```bash
  python3 gcg/llm-attacks/gcg.py
  ```
  So the results go in the right results subfolder.
- The ```paper/gcg/llm-attacks/my_experiments``` folder which contains python scripts for some the experiments mentionned in my internship report.
  - The ```one_example``` folder contains the scripts to run autoprompt, gcg and our method on a single example. They need to be run from the ```paper/gcg/llm-attacks/my_experiments``` folder
    ```bash
    python3 one_example/[autoprompt or gcg or our_method].py
    ```
  - The ```attraction_test.py``` file aims to check if we can noise an input embedding such that the LLM cannot answer it corectly but can after optimizing the input with the loss attracting the embeddings toward the embedding matrix. You can run it like this:
    ```bash
    python3 attraction_test.py
    ```
  - The ```autoprompt.py``` file defines a function that runs autoprompt on one example (adapted from the gcg code).
  - The ```entropy_test.py``` file checks two implementation of the entropy. You can run it like this:
    ```bash
    python3 entropy_test.py
    ```
  - The ```gcg.py``` file defines a function that runs gcg on one example.
  - The ```launch_nbsteps_comparison.py``` file runs the prompt_tuning, our_method, gcg and autoprompt methods on a dataset. Each method is ran until it succeeds and we save the number of steps it took to succedd. We can then average on the dataset. You need to run it in the ```paper/gcg/llm-attacks/my_experiments``` folder:
    ```bash
    python3 launch_nbsteps_comparison.py
    ```
  - The ```launch_success_comparison.py``` file runs the prompt_tuning, our_method, gcg and autoprompt methods on a dataset. Each method is ran for a fixed number of steps. For each example of the dataset we save if the moethod succeeded or not and then we can get an average success rate on the dataset. You need to run it in the ```paper/gcg/llm-attacks/my_experiments``` folder:
    ```bash
    python3 launch_success_comparison.py
    ```
  - The ```my_attack.py``` file contains some losses and the function that computes gradient of the loss with respect to the prompt.
  - The ```one_prompt_tuning.py``` file corresponds to manual prompt tuning with the gradient computation function from the llm-attacks code. You need to run it in the ```paper/gcg/llm-attacks/my_experiments``` folder:
    ```bash
    python3 one_prompt_tuning.py
    ```
  - The ```our_method.py``` file defines a function that runs our method on one example. The scripts using this function need to be run in the ```paper/gcg/llm-attacks/my_experiments``` folder.
  - The ```prompt_tuning.py``` file defines a function that runs manual prompt tuning on one example.
  - The ```run_one.py``` file runs our method on one example. You need to run it in the ```paper/gcg/llm-attacks/my_experiments``` folder:
    ```bash
    python3 run_one.py
    ```
  - The ```utils.py``` file contains some utility functions such as checking if a prompt achieves a successful attack or measuring distance of embeddings to the embedding matrix.
