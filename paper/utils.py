import os
import torch
import torch.nn as nn
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

def tokenize_input(tokenizer, input, device):
    # we're using a chat model on some question answer dataset => need to apply a chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    return tokenized_chat

# CROSS-ENTROPY LOSS
def target_loss(model, tokenizer, query, target):
    # return -log(p(label|input))
    query_ids = tokenizer(query, add_special_tokens=False, return_tensors='pt')['input_ids'].to(model.device)
    #query_ids = tokenize_input(tokenizer, query, model.device)
    target_ids = tokenizer(target, add_special_tokens=False, return_tensors='pt')['input_ids'].to(model.device)
    ids = torch.cat((query_ids, target_ids), dim=-1)

    outputs = model(input_ids=ids)
    logits = outputs.logits

    query_length = query_ids.shape[1]
    target_length = target_ids.shape[1]
    input_length = ids.shape[1]

    for index in range(query_length-1, input_length-1):
        print(tokenizer.decode(ids[0, index]), " -> ", tokenizer.decode(torch.argmax(logits[0, index, :], dim=-1)), " with probability ", torch.max(F.softmax(logits[0, index, :], dim=-1)).item())

    query_outputs = model(input_ids=query_ids)
    query_logits = query_outputs.logits
    print("most probable next token after", tokenizer.decode(query_ids[0, -1]), ":", tokenizer.decode(torch.argmax(query_logits[0, -1, :], dim=-1)), "with probability", torch.max(F.softmax(query_logits[0, -1, :], dim=-1)).item())

    target_slice = slice(input_length-target_length, input_length)
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)

    # same as gcg ----------------
    crit = nn.CrossEntropyLoss(reduction='none')

    #print("target_slice text:", tokenizer.decode(ids[0,target_slice]))
    #print("loss_slice text:", tokenizer.decode(ids[0,loss_slice]))

    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    return loss.mean(dim=-1)

"""
def compute_loss(model, tokenizer, input_batch, labels):
    # return -log(p(label|input))
    device = model.device
    batch_size = len(input_batch)
    inputs = [input_batch[i] + labels[i] for i in range(batch_size)]
    input_ids = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
    label_ids = tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
    print("label_ids.shape:", label_ids.shape)
    label_length = label_ids.shape[1]
    outputs = model(input_ids=input_ids)
    print("outputs.logits.shape:", outputs.logits.shape)
    distributions = F.softmax(outputs.logits, dim=-1) # [batch_size, seq_len, vocab_size], seq_len = n+m

    # for each batch, for each token in the sequence we print the 3 most probable tokens and their probabilities
    for i in range(batch_size):
        #print("input_ids:", tokenizer.decode(input_ids[i, :]))
        #print("greedy decoded output:", tokenizer.decode(torch.argmax(distributions[i, :, :], dim=-1)))
        print("Greedy predictions:")
        for index in range(len(input_ids[i, :])):
            print(tokenizer.decode(input_ids[i, index]), " -> ", tokenizer.decode(torch.argmax(distributions[i, index, :], dim=-1)), " with probability ", torch.max(distributions[i, index, :]).item())

        print("\nSelection of the label tokens and probabilities:")
        print("label input_ids[i, -label_length:]:", tokenizer.decode(input_ids[i, -label_length:]))
        print("input_ids[i, -(label_length+1):]:", tokenizer.decode(input_ids[i, -(label_length+1):-1]))
        print("intersting probabilities distrib[-(label_length+1):-1]:", torch.max(distributions[i, -(label_length+1):-1, :], dim=-1))
    
    selected_distributions = distributions[:, -(label_length+1):-1, :] # [batch_size, m, vocab_size]
    #selected_distributions = distributions[:, -label_length:, :] # [batch_size, m, vocab_size]

    # for each batch, for each token in the prompt, we want to get the probability of the corresponding token in the label
    probabilities = torch.zeros((batch_size, label_length)).to(device)
    for batch in range(batch_size):
        print("selected label tokens:")
        for i in range(label_length):
            probabilities[batch, i] = selected_distributions[batch, i, label_ids[batch, i]]
            print(tokenizer.decode(label_ids[batch, i]), "which has probability", selected_distributions[batch, i, label_ids[batch, i]])
    loss = -torch.mean(torch.log(probabilities), dim=-1).mean() # and mean over the batch
    #loss = -torch.sum(torch.log(probabilities), dim=-1).mean() # and mean over the batch
    return loss
"""

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
    """
    ask1_txt = "Translate \""
    ask2_txt = "\" to english. Sure! Here is the translation: "
    """
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
    input_ids = tokenizer(input, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(device)
    #input_ids = tokenize_input(tokenizer, input, device)
    #print("input_ids:", tokenizer.decode(input_ids[0]))

    # debug
    """
    print("input_ids before inference:", tokenizer.decode(input_ids['input_ids'][0]))
    outputs = model(input_ids=input_ids['input_ids'])
    logits = outputs.logits
    print("most probable next token before inference:", tokenizer.decode(torch.argmax(logits[0, -1, :], dim=-1)), "with probability", torch.max(F.softmax(logits[0, -1, :], dim=-1)).item())
    """
    
    model_outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    #model_outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
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

def check_for_attack_success_noref_with_target(model, tokenizer, input, target, test_prefixes, max_new_tokens):
    device = model.device
    input_ids = tokenizer(input+target, return_tensors="pt", return_token_type_ids=False).to(device)
    model_outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    jailbroken = not any([prefix in text_output for prefix in test_prefixes])
    return text_output, jailbroken

# TRAINING
def individual_training(model, tokenizer, input, target, num_epochs, optimizer, until_success=False, test_prefixes=None):
    model.train()
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        #loss = compute_loss(model, tokenizer, [input], [target]) # (bacth_size = 1)
        loss = target_loss(model, tokenizer, input, target)
        loss.backward()
        optimizer.step()
    
        pbar.set_postfix({'loss': loss.item()})

        if until_success:
            jailbroken = check_for_attack_success_noref(model, tokenizer, input, target, test_prefixes)
            if jailbroken:
                return epoch
            
    return loss.item()

# WHILE TRAINING
"""
def while_individual_training(model, tokenizer, input, target, max_num_epochs, epsilon, optimizer, until_success=False, test_prefixes=None):
    model.train()
    pbar = tqdm(range(max_num_epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        loss = compute_loss(model, tokenizer, [input], [target]) # (bacth_size = 1)
        loss.backward()
        optimizer.step()
    
        pbar.set_postfix({'loss': loss.item()})

        if loss.item() < epsilon:
            return loss.item()
        
    return loss.item()
"""

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
