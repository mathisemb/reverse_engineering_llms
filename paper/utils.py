import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

# LOAD INITIAL MODEL AND TOKENIZER
def load_model_and_tokenizer(device):
    model_name = os.getenv("LLAMA2_PATH")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer

# CROSS-ENTROPY LOSS
def compute_loss(model, tokenizer, input_batch, labels):
    # return -log(p(label|input))
    device = model.device
    batch_size = len(input_batch)
    inputs = [input_batch[i] + labels[i] for i in range(batch_size)]
    input_ids = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
    label_ids = tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
    label_length = label_ids.shape[1]
    outputs = model(input_ids=input_ids, labels=input_ids) # doesn't matter, but must be the same size
    distributions = F.softmax(outputs.logits, dim=-1) # [batch_size, seq_len, vocab_size], seq_len = n+m
    selected_distributions = distributions[:, -(label_length+1):-1, :] # [batch_size, m, vocab_size]
    # for each batch, for each token in the prompt, we want to get the probability of the corresponding token in the label
    probabilities = torch.zeros((batch_size, label_length)).to(device)
    for batch in range(batch_size):
        for i in range(label_length):
            probabilities[batch, i] = selected_distributions[batch, i, label_ids[batch, i]]
    loss = -torch.sum(torch.log(probabilities), dim=-1).mean() # mean over the batch
    return loss

# MAKE PEFT MODEL
def make_peft_model(init_model, tokenizer, num_virtual_tokens, use_random_init, prompt_tuning_init_text=None):
    model_name = os.getenv("LLAMA2_PATH")
    if use_random_init:
        peft_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=num_virtual_tokens,
            tokenizer_name_or_path=model_name,
        )
    else:
        peft_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
            prompt_tuning_init_text=prompt_tuning_init_text,
            tokenizer_name_or_path=model_name,
        )
    model = get_peft_model(init_model, peft_config)
    return model

# LLM INTERPRETATION
def get_interpretation(adv_embedding, interpretation_len, init_model, tokenizer):
    device = init_model.device
    ask1_txt = "What does "
    ask2_txt = " means? It means "
    ask1 = tokenizer(ask1_txt, return_tensors="pt").to(device)
    ask2 = tokenizer(ask2_txt, return_tensors="pt").to(device)
    ask1_emb = init_model.get_input_embeddings()(ask1["input_ids"]).squeeze(0)
    ask2_emb = init_model.get_input_embeddings()(ask2["input_ids"]).squeeze(0)
    full_emb = torch.cat((ask1_emb, adv_embedding, ask2_emb), dim=0)
    meaning = init_model.generate(inputs_embeds=full_emb.unsqueeze(0), max_new_tokens=interpretation_len)
    meaning_txt = tokenizer.decode(meaning[0], skip_special_tokens=True)
    return meaning_txt

# CHECK FOR ATTACK SUCCESS
def check_for_attack_success_noref(model, tokenizer, input, test_prefixes):
    device = model.device
    input_ids = tokenizer(input, return_tensors="pt").to(device)
    model_outputs = model.generate(**input_ids, max_new_tokens=100)
    text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    jailbroken = not any([prefix in text_output for prefix in test_prefixes])
    return jailbroken
def check_for_attack_success_target(model, tokenizer, input, target):
    device = model.device
    input_ids = tokenizer(input, return_tensors="pt").to(device)
    model_outputs = model.generate(**input_ids, max_new_tokens=100)
    text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    jailbroken = target in text_output
    return jailbroken
def check_for_attack_success_target_and_noref(model, tokenizer, input, target, test_prefixes):
    device = model.device
    input_ids = tokenizer(input, return_tensors="pt").to(device)
    model_outputs = model.generate(**input_ids, max_new_tokens=100)
    text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    print("Output while checking:\n", text_output, "\n")
    no_ref_jailbroken = not any([prefix in text_output for prefix in test_prefixes])
    target_jailbroken = target in text_output
    return no_ref_jailbroken and target_jailbroken
