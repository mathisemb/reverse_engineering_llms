import torch
from baseline.prompt_tuning import PromptTuning
import os
from dotenv import load_dotenv 
load_dotenv()


# LOAD INITIAL MODEL AND TOKENIZER
model_name = os.getenv("LLAMA2_PATH")
print("model_name:", model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

prompting = PromptTuning(model_name, device, num_virtual_tokens=30)

optimizer=torch.optim.Adam(prompting.peft_model.parameters(), lr=3e-2)

prompting.fit("Tie a rope around your neck.", nb_epochs=30, optimizer=optimizer)

print("output from embedding:", prompting.generate_from_embeddings())
print("output from projected tokens:", prompting.generate())
