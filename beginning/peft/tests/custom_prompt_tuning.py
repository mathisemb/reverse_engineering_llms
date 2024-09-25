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
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.RANDOM,
    num_virtual_tokens=10,
    tokenizer_name_or_path="bigscience/bloomz-560m",
)
model = get_peft_model(init_model, peft_config)
model.print_trainable_parameters()


# LOG LIKELIHOOD OF A SEQUENCE OF EMBEDDINGS
def log_likelihood(embeddings_sentence, temperature=0.001):
    # embeddings_sentence.shape = [batch_size, seq_len, embedding_dim]
    bos_embedding = init_model.get_input_embeddings().weight[tokenizer.bos_token_id].unsqueeze(0).unsqueeze(0) # [1, 1, embedding_dim]
    modified_embeddings_sentence = torch.cat((bos_embedding, embeddings_sentence[:, :-1, :]), dim=1) # [1, seq_len, embedding_dim]
    outputs = init_model(inputs_embeds=modified_embeddings_sentence)
    probalities = torch.softmax(outputs.logits, dim=-1) # [batch_size, seq_len, vocab_size]
    embedding_matrix = init_model.get_input_embeddings().weight # [vocab_size, embedding_dim]
    WEx = torch.matmul(embedding_matrix, embeddings_sentence.squeeze(0).T) # [vocab_size, seq_len]
    
    use_argmax = False # Cannot do this because argmax is not differentiable:
    if use_argmax:
        indices = torch.argmax(WEx, dim=0) # [seq_len]
        cond_prob = torch.tensor([probalities[:,i,indices[i]] for i in range(indices.shape[0])]) # [seq_len]
    else:
        d = torch.softmax(WEx.T/temperature, dim=-1) # [seq_len, vocab_size]
        A = d*(probalities.squeeze(0)) # element wise multiplication [seq_len, vocab_size]
        cond_prob = torch.sum(A, dim=-1) # [seq_len]
    
    log_likelihood = torch.sum(torch.log(cond_prob))
    return log_likelihood

# Tests
token_ids = tokenizer("Hello,", return_tensors='pt')['input_ids'].to(device)
embeddings = init_model.get_input_embeddings().weight[token_ids]
print("likelihood(Hello,):", log_likelihood(embeddings).item())

token_ids = tokenizer("My name is John", return_tensors='pt')['input_ids'].to(device)
embeddings = init_model.get_input_embeddings().weight[token_ids]
print("likelihood(My name is John):", log_likelihood(embeddings).item())

token_ids = tokenizer("John my is name", return_tensors='pt')['input_ids'].to(device)
embeddings = init_model.get_input_embeddings().weight[token_ids]
print("likelihood(John my is name):", log_likelihood(embeddings).item())

random_embeddings = torch.randn(1, 4, init_model.config.hidden_size).to(device)
print("likelihood(random_embeddings):", log_likelihood(random_embeddings).item())


# MEAN DISTANCE TO EMBEDDING MATRIX
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


# TRAINING
# can also be done with the Transformers Trainer class
# https://huggingface.co/transformers/main_classes/trainer.html

def training(input, label, num_epochs, optimizer, alpha=1, beta=1, gamma=1, use_likelihood=True, use_entropy=True):
    # before training
    print("mean_distance_to_embedding_matrix BEFORE:", mean_distance_to_embedding_matrix(model))
    print("likelihood of the optimized prompt BEFORE:", log_likelihood(model.get_prompt(batch_size=1)).item())

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0

        input_ids = tokenizer.encode(input, return_tensors='pt').to(device)
        labels = tokenizer.encode(label, return_tensors='pt').to(device)
        outputs = model(input_ids=input_ids, labels=labels)

        # save the original loss
        loss = outputs.loss
        total_loss += alpha * loss.detach().float()

        # CUSTOMIZED LOSS
        prompt_embeddings = model.get_prompt(batch_size=1)

        if use_likelihood: # likelihood of the optimized prompt
            total_loss += -beta*log_likelihood(prompt_embeddings)

        if use_entropy: # entropy of the softmax of the dot product between embedding matrix and the embeddings being optimized
            embedding_matrix = model.get_input_embeddings().weight
            dot_product = torch.matmul(prompt_embeddings.squeeze(0), embedding_matrix.T)
            softmax_dot_product = F.softmax(dot_product, dim=-1)
            entropy = -torch.sum(softmax_dot_product * torch.log(softmax_dot_product), dim=-1).mean()
            total_loss += gamma*entropy
        
        """
        For example
        loss: 10.429055213928223
        likelihood: 191.89453125
        entropy: 12.365477561950684
        """
        
        total_loss.backward() # backward on the total loss
        optimizer.step()
        optimizer.zero_grad()
    
    # after training
    print("mean_distance_to_embedding_matrix AFTER:", mean_distance_to_embedding_matrix(model))
    print("likelihood of the optimized prompt AFTER:", log_likelihood(model.get_prompt(batch_size=1)))

training(input="London is the capital of",
         label="is the capital of France",
         num_epochs=100,
         optimizer=torch.optim.Adam(model.parameters(), lr=3e-2),
         alpha=20,
         beta=1,
         gamma=10,
         use_likelihood=True,
         use_entropy=True)


# OPTIMIZED PROMPT
prompt_embeddings = model.get_prompt(batch_size=1)
print("prompt_embeddings.shape:", prompt_embeddings.shape) # [batch_size, num_tokens, embedding_dim] = [1, 10, 1024]
embedding_matrix = model.get_input_embeddings().weight.data
nearest_tokens = torch.argmax(torch.matmul(prompt_embeddings, embedding_matrix.T), dim=-1) # Find the nearest tokens in the embedding space
decoded_prompt = tokenizer.decode(nearest_tokens[0], skip_special_tokens=True)
print("decoded_prompt:", decoded_prompt)


# INFERENCE TEST
input = "London is the capital of"
input_ids = tokenizer(input, return_tensors="pt").to(device)
model_outputs = model.generate(**input_ids, max_new_tokens=5)
print("model output with optimized embeddings:", tokenizer.decode(model_outputs[0], skip_special_tokens=True))

# INFERENCE WITH THE CLOSEST TOKENS AND THE ORIGINAL MODEL
prompt_ids = tokenizer(decoded_prompt+input, return_tensors="pt").to(device)
model_outputs = init_model.generate(**prompt_ids, max_new_tokens=5)
print("model_outputs with closest optimized tokens:", tokenizer.decode(model_outputs[0], skip_special_tokens=True))
