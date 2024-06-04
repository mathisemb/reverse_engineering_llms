from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
from peft import PrefixTuningConfig, get_peft_model
from tqdm import tqdm
from peft import PromptEncoderConfig, get_peft_model


# MODEL AND TOKENIZER
model_name = "bigscience/bloomz-560m"
#model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" #error with past_key_value

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


# PEFT MODEL
peft_config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, encoder_hidden_size=128)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# TRAINING
# can also be done with the Transformers Trainer class
# https://huggingface.co/transformers/main_classes/trainer.html
lr = 3e-2
num_epochs = 100
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


# OPTIMIZED PROMPT
prompt_embeddings = model.get_prompt(batch_size=1)
print("prompt_embeddings:", prompt_embeddings.shape) # [batch_size, num_tokens, embedding_dim] = [1, 20, 1024]
embedding_matrix = model.get_input_embeddings().weight.data
print("embedding_matrix:", embedding_matrix.shape) # [vocab_size, embedding_dim]
nearest_tokens = torch.argmax(torch.matmul(prompt_embeddings, embedding_matrix.T), dim=-1) # Find the nearest tokens in the embedding space
decoded_prompt = tokenizer.decode(nearest_tokens[0], skip_special_tokens=True)
print("decoded_prompt:", decoded_prompt)


# INFERENCE TEST
input = "London is the capital of"
input_ids = tokenizer(input, return_tensors="pt").to(device)
model_outputs = model.generate(**input_ids, max_new_tokens=5)
print("model_outputs:", tokenizer.decode(model_outputs[0], skip_special_tokens=True))
