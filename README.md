# Reverse-engineering LLMs: making the most of the context

This repository contains the work I did during my [MVA](https://www.master-mva.com/) research internship. The internship took place in the [MILES team](https://www.lamsade.dauphine.fr/wp/miles/) of LAMSADE, Université Paris-Dauphine, PSL University. This team is oriented toward trustworthy and explainable machine learning. I was advised by Prof. Alexandre Allauzen and his PhD student Florian Le Bronnec.

## Context
Large Language Models (LLMs) display notable variations in performance based on the nature of in-context prompts. This performance discrepancy is particularly evident when tasks are vaguely described, resulting in poorer outcomes compared to scenarios where prompts include specific examples, called few-shots in-context learning. Despite the significance of prompt influence on LLMs’ performance, this aspect remains relatively unexplored.

For more details on the initial goal of the internship, see ```subject.pdf```.

## What I did
I focused on reverse-engineering LLMs which means finding the input that makes the LLM generate a target output, for example through an optimization process. We choosed to work with an adversarial attack setup. Given a harmful target we need to find an adversarial prefix to the query so the LLM generate the answer despite its alignement.

For more details, see my intership report ```report.pdf``` and my conclusion ```conclusion.pdf```.

## Repository overview
- The ```beginning``` folder contains my first tries. You can ignore it (or maybe it's interesting to check my FluentPrompt implementation in ```beginning/baseline/fluent_prompt.py``` as the authors did not provided the code nor responded to my mail).
- The ```experiments``` folder contains the code of the experiments presented in my internship report. You can find a specific README in the folder.
- The ```meetings``` folder contains small reports of some ideas I came up with throughout the internship. If you want a complete report, see ```report.pdf```.
- The ```paper``` folder contains the experiments I did during the month after submitting my report. You can find a specific README in the folder.
