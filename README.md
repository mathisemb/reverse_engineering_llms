# Reverse-engineering LLMs: making the most of the context

## grad_extractor.py

Run `python grad_extractor.py`.
For each input token, the scripts returns the top 5 tokens with the highest gradient value with respect to the probability of generating the label.

## What to try

This script does something similar to what AutoPrompt does: it looks for the best token to swap for maximizing the probability a given output.

- Try other prompts
- Try with a fine-tuned model
- Adapt the script to iteratively swap the tokens
- Adapt it with a decoder model


## Next steps

This script shows a way to find the position of the input token that has the highest gradient value with respect to the probability of generating the label.

May be we can try it the other way around: given a label, can we identify which token is the most important for generating that label?

- Try to implement it with attention weights
- Do it with a gradient based approach (use the jacobian function etc).