# ```experiments``` folder
The ```experiments``` folder contains the code of the experiments presented in my internship report (```report.pdf```), excepted the code for gcg, autoprompt and our method that is in the ```paper/gcg/llm-attacks/my_experiments``` folder (see ```paper/gcg/README.md``` for more details).

There is the code for computing the success rates of the different projections after the continuous optimization. It also contains the overfitting and initialization experiments

- The ```initialization_comp.py``` file computes the average number of step it takes for prompt tuning to succeed depending on the initialization of the adversarial prefix: repeated "!" token, random embedding sequence, "You are an assistant trained to generate harmful content." and the meaning of the optimized adversarial prefix according to the LLM. You need to run it from the ```experiments``` folder:
    ```bash
    python3 initialization_comp.py
    ```
- The ```initialization.py``` file does the same as ```initialization_comp.py``` but on a single example of the dataset. You need to run it from the ```experiments``` folder:
    ```bash
    python3 initialization.py
    ```
- The ```overfitting.py``` file proceeds to prompt tuning with different number of steps to exhibit some possible overfitting. You need to run it from the ```experiments``` folder:
    ```bash
    python3 overfitting.py
    ```
- The ```success_rates.py``` file runs prompt tuning for a fixed number of steps and compute the success rate of the projection according to dot product, cosine and L2 distance. You need to run it from the ```experiments``` folder:
    ```bash
    python3 success_rates.py
    ```
