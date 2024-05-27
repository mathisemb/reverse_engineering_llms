from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
import torch.nn.functional as F


# MODEL AND TOKENIZER
model_name = "bigscience/bloomz-560m"
#model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" #error with past_key_value

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

init_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


# PEFT MODEL
#prompt_tuning_init_text = "Classify if the tweet is a complaint or no complaint.\n"
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.RANDOM, #.TEXT,
    #num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
    num_virtual_tokens=20,
    #prompt_tuning_init_text=prompt_tuning_init_text,
    tokenizer_name_or_path="bigscience/bloomz-560m",
)
model = get_peft_model(init_model, peft_config)
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

"""
def compute_likelihood(embeddings_sentence):
    model.eval()
    with torch.no_grad():
        outputs = init_model(inputs_embeds=embeddings_sentence)
        logits = outputs.logits[:, :-1, :].squeeze(0) # Exclude the last token (prediction is conditional on previous)
        distribution = torch.softmax(logits, dim=-1)
        # embeddings_sentence closest tokens
        closest_tokens = torch.argmax(embeddings_sentence.squeeze(0), dim=-1)
        probabilities = torch.Tensor([distribution[i, token] for i, token in enumerate(inputs.squeeze(0))])
        print("likelihood:", likelihood.shape) # [num_tokens, vocab_size]
        #likelihoods = likelihood[torch.arange(len(embeddings_sentence)), inputs.squeeze(0)]
        sentence_likelihood = torch.prod(likelihood).item()#torch.prod(likelihoods).item()
    return sentence_likelihood
"""

def mean_distance_to_embedding_matrix(model):
    # returns the mean of cosine sim between each embeddings and their closest (cosine) embedding in the embedding matrix
    prompt_embeddings = model.get_prompt(batch_size=1)[0] # [nb_embeddings, embedding_dim]
    embedding_matrix = model.get_input_embeddings().weight # [vocab_size, embedding_dim]
    dot_product = torch.matmul(prompt_embeddings, embedding_matrix.T)
    denom = torch.matmul(torch.norm(prompt_embeddings, dim=-1, keepdim=True), torch.norm(embedding_matrix, dim=-1, keepdim=True).T)
    cosine_sim = dot_product/denom
    one_minus_cosine_sim = torch.ones_like(cosine_sim) - cosine_sim
    mins = torch.min(one_minus_cosine_sim, dim=-1).values
    return mins.mean().item()

def training(input, label, num_epochs, alpha=1, beta=1, use_likelihood=True, use_entropy=True):
    # before training
    print("mean_distance_to_embedding_matrix BEFORE:", mean_distance_to_embedding_matrix(model))

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0

        input_ids = tokenizer.encode(input, return_tensors='pt').to(device)
        labels = tokenizer.encode(label, return_tensors='pt').to(device)
        outputs = model(input_ids=input_ids, labels=labels)

        # save the original loss
        loss = outputs.loss
        total_loss += loss.detach().float()

        # CUSTOMIZED LOSS
        prompt_embeddings = model.get_prompt(batch_size=1)

        """
        if use_likelihood: # likelihood of the optimized prompt
            likelihood = compute_likelihood(prompt_embeddings)
            total_loss += -alpha*likelihood
        """

        if use_entropy: # entropy of the softmax of the dot product between embedding matrix and the embeddings being optimized
            embedding_matrix = model.get_input_embeddings().weight
            dot_product = torch.matmul(prompt_embeddings.squeeze(0), embedding_matrix.T)
            softmax_dot_product = F.softmax(dot_product, dim=-1)
            entropy = -torch.sum(softmax_dot_product * torch.log(softmax_dot_product), dim=-1).mean()
            total_loss += beta*entropy

        total_loss.backward() # backward on the total loss
        #loss.backward() # backward on the original loss
        optimizer.step()
        optimizer.zero_grad()
    
    # after training
    print("mean_distance_to_embedding_matrix AFTER:", mean_distance_to_embedding_matrix(model))

training(input, label, num_epochs, beta=10, use_likelihood=False, use_entropy=True)


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
