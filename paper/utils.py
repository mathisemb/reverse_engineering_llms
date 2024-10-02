import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from tqdm import tqdm

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
    """
    ask1_txt = "What does "
    ask2_txt = " means? It means "
    """
    ask1_txt = "Translate \""
    ask2_txt = "\" to english. Sure! Here is the translation: "
    ask1 = tokenizer(ask1_txt, return_tensors="pt").to(device)
    ask2 = tokenizer(ask2_txt, return_tensors="pt").to(device)
    ask1_emb = init_model.get_input_embeddings()(ask1["input_ids"]).squeeze(0)
    ask2_emb = init_model.get_input_embeddings()(ask2["input_ids"]).squeeze(0)
    full_emb = torch.cat((ask1_emb, adv_embedding, ask2_emb), dim=0)
    meaning = init_model.generate(inputs_embeds=full_emb.unsqueeze(0), max_new_tokens=interpretation_len)
    meaning_txt = tokenizer.decode(meaning[0], skip_special_tokens=True)
    return meaning_txt

# CHECK FOR ATTACK SUCCESS
def check_for_attack_success_noref(model, tokenizer, input, test_prefixes, max_new_tokens):
    device = model.device
    input_ids = tokenizer(input, return_tensors="pt").to(device)
    model_outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    jailbroken = not any([prefix in text_output for prefix in test_prefixes])
    return text_output, jailbroken
def check_for_attack_success_target(model, tokenizer, input, target, max_new_tokens):
    device = model.device
    input_ids = tokenizer(input, return_tensors="pt").to(device)
    model_outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    jailbroken = target in text_output
    return text_output, jailbroken
def check_for_attack_success_target_and_noref(model, tokenizer, input, target, test_prefixes, max_new_tokens):
    device = model.device
    input_ids = tokenizer(input, return_tensors="pt").to(device)
    model_outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    no_ref_jailbroken = not any([prefix in text_output for prefix in test_prefixes])
    target_jailbroken = target in text_output
    jailbroken = no_ref_jailbroken and target_jailbroken
    return text_output, jailbroken

# TRAINING
def individual_training(model, tokenizer, input, target, num_epochs, optimizer, until_success=False, test_prefixes=None):
    model.train()
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        loss = compute_loss(model, tokenizer, [input], [target]) # (bacth_size = 1)
        loss.backward()
        optimizer.step()
    
        pbar.set_postfix({'loss': loss.item()})

        if until_success:
            jailbroken = check_for_attack_success_noref(model, tokenizer, input, target, test_prefixes)
            if jailbroken:
                return epoch

# CLOSEST TOKEN
def closest_token(embeddings, model, metric='cosine'):
    # return the token embedding of the embedding matrix that is the closest to the input embedding
    # metric is in ['dot product', 'cosine', 'L2']
    # embeddings: [seq_len, embedding_dim]
    embedding_matrix = model.get_input_embeddings().weight # [vocab_size, embedding_dim]
    dot_product = torch.matmul(embeddings, embedding_matrix.T) # [seq_len, vocab_size]

    seq_len = embeddings.shape[0]
    vocab_size = embedding_matrix.shape[0]

    if metric == 'dot product':
        return torch.argmax(dot_product, dim=-1)
    elif metric == 'L2':
        closest_embeddings = []
        for i in range(seq_len):
            closest_embeddings.append(torch.argmin(torch.norm(embeddings[i] - embedding_matrix, p=2, dim=-1)))
        return torch.tensor(closest_embeddings)
    elif metric == 'cosine':
        emb_norms = torch.norm(embeddings, dim=-1, keepdim=True)
        emb_norms = emb_norms.repeat(1, vocab_size)
        mat_norms = torch.norm(embedding_matrix, dim=-1, keepdim=True).T
        mat_norms = mat_norms.repeat(seq_len, 1)
        norms = emb_norms*mat_norms
        cosine_similarity = dot_product / norms
        closest_embeddings = torch.argmax(cosine_similarity, dim=-1)
        return closest_embeddings
