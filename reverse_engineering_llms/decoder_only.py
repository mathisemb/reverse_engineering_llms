import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


# MODEL AND TOKENIZER
#model_name = "bigscience/bloomz-560m"
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


# INFERENCE TEST
input = "London is the capital of"
input_ids = tokenizer(input, return_tensors="pt").to(device)
model_outputs = model.generate(**input_ids, max_new_tokens=5)
print("model_outputs:", tokenizer.decode(model_outputs[0], skip_special_tokens=True))


# TRAINING TEST
input = "London is the capital of"
label = "is the capital of France"
#label = "France" # doesn't work with decoder-only
input_ids = tokenizer.encode(input, return_tensors='pt').to(device)
labels = tokenizer.encode(label, return_tensors='pt').to(device)
outputs = model(input_ids=input_ids, labels=labels)
print("outputs.logits:", outputs.logits.shape)
print("After 3rd word prediction:", tokenizer.decode(torch.argmax(outputs.logits[0,3])))
print("After last word prediction:", tokenizer.decode(torch.argmax(outputs.logits[0,-1])))


# TEST OF THE LOSS
input_for_true = "London is the capital of the United"
true_label = "is the capital of the United Kingdom"
input_for_wrong = "London is the capital of"
wrong_label = "is the capital of France"
input_for_true_ids = tokenizer.encode(input_for_true, return_tensors='pt').to(device)
input_for_wrong_ids = tokenizer.encode(input_for_wrong, return_tensors='pt').to(device)
true_labels = tokenizer.encode(true_label, return_tensors='pt').to(device)
wrong_labels = tokenizer.encode(wrong_label, return_tensors='pt').to(device)
true_outputs = model(input_ids=input_for_true_ids, labels=true_labels)
wrong_outputs = model(input_ids=input_for_wrong_ids, labels=wrong_labels)
print("Loss for true label:", true_outputs.loss.item())
print("Loss for wrong label:", wrong_outputs.loss.item())
