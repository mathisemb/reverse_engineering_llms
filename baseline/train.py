import torch
from baseline.prompt_tuning import PromptTuning
from baseline.autoprompt import AutoPrompt
from baseline.new_method import NewMethod
from baseline.fluent_prompt import FluentPrompt
from baseline.forward_projection import ForwardProjection
import os
from dotenv import load_dotenv 
load_dotenv()

# LOAD INITIAL MODEL AND TOKENIZER
model_name = os.getenv("LLAMA2_PATH")
#model_name = os.getenv("MISTRAL_PATH")
print("model_name:", model_name)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# AutoPrompt
"""
prompting = AutoPrompt(model_name, device, num_virtual_tokens=5)
prompting.fit("France", nb_epochs=10, k=3, approximation=False)
print("prompt:", prompting.get_prompt())
print("output after virtual tokens:", prompting.generate(max_length=100))
"""

# NewMethod
"""
prompting = NewMethod(model_name, device, num_virtual_tokens=20)
optimizer = torch.optim.Adam(prompting.peft_model.parameters(), lr=3e-2)
prompting.fit(
    target="Tie a rope around your neck.",
    nb_epochs=20,
    optimizer=optimizer,
    alpha=1,
    beta=1,
    gamma=1)
print("output after virtual tokens:", prompting.generate())
"""

# FluentPrompt
"""
prompting = FluentPrompt(model_name, device, num_virtual_tokens=30)
optimizer = torch.optim.Adam(prompting.peft_model.parameters(), lr=3e-2)
prompting.fit(
    target="Hello.",
    nb_epochs=100,
    optimizer=optimizer,
    l_task=0.5,
    l_fluency=0.5)
print("output after virtual tokens:", prompting.generate())
"""

# PromptTuning
"""
print("---PromptTuning---")
prompting = PromptTuning(model_name, device, num_virtual_tokens=30)
optimizer = torch.optim.Adam(prompting.peft_model.parameters(), lr=3e-2)
prompting.fit(
    target="Hello.",
    nb_epochs=100,
    optimizer=optimizer)
print("prompt:", prompting.get_prompt())
print("output from embedding:", prompting.generate_from_embeddings())
print("output from projected tokens:", prompting.generate())
"""

# ForwardProjection
"""
print("\n---ForwardProjection---")
prompting = ForwardProjection(model_name, device, num_virtual_tokens=30)
optimizer = torch.optim.Adam(prompting.peft_model.parameters(), lr=3e-2)
prompting.fit(
    target="Hello.",
    nb_epochs=100,
    optimizer=optimizer)
print("prompt:", prompting.get_prompt())
print("output from embedding:", prompting.generate_from_embeddings())
print("output from projected tokens:", prompting.generate())
"""
