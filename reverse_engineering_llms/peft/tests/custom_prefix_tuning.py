from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
from peft import PrefixTuningConfig, get_peft_model
from tqdm import tqdm


# MODEL AND TOKENIZER
model_name = "bigscience/bloomz-560m"
#model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" #error with past_key_value

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


# PEFT PREFIX TUNING MODEL
peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=3)
model = get_peft_model(model, peft_config).to(device)
model.print_trainable_parameters() # For example: "trainable params: 983,040 || all params: 560,197,632 || trainable%: 0.1754809274167014"


# TRAINING
# can also be done with the Transformers Trainer class
# https://huggingface.co/transformers/main_classes/trainer.html
lr = 3e-2
num_epochs = 10
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

input = "London is the capital of"
label = "is the capital of France"

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0

    input_ids = tokenizer.encode(input, return_tensors='pt').to(device)
    labels = tokenizer.encode(label, return_tensors='pt').to(device)
    outputs = model(input_ids=input_ids, labels=labels)
    
    loss = outputs.loss
    total_loss += loss.detach().float()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#print("Optimized prompt:", model.get_prompt())

# INFERENCE TEST
input = "London is the capital of"
input_ids = tokenizer(input, return_tensors="pt").to(device)
model_outputs = model.generate(**input_ids, max_new_tokens=5)
print("model_outputs:", tokenizer.decode(model_outputs[0], skip_special_tokens=True))
