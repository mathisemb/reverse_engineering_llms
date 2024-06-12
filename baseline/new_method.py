import torch
from baseline.prompting import Prompting
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
import torch.nn.functional as F
from tqdm import tqdm

class NewMethod(Prompting):
    """
    for new method we use peft and

    self.model corresponds to the original model
    self.peft_model corresponds to the augmented peft model
    self.prompt corresponds to the virtual embeddings projected into the vocabulary and converted to a string
    """

    def __init__(self, model_name, device, num_virtual_tokens):
        super().__init__(model_name, device)
        peft_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=num_virtual_tokens,
            tokenizer_name_or_path=model_name,
        )
        self.peft_model = get_peft_model(self.model, peft_config)
    
    def compute_entropy(self, prompt_embeddings):
        embedding_matrix = self.peft_model.get_input_embeddings().weight
        dot_product = torch.matmul(prompt_embeddings.squeeze(0), embedding_matrix.T)
        softmax_dot_product = F.softmax(dot_product, dim=-1)
        entropy = -torch.sum(softmax_dot_product * torch.log(softmax_dot_product), dim=-1).mean()
        return entropy

    def log_likelihood(self, embeddings_sentence):
        # embeddings_sentence.shape = [batch_size, seq_len, embedding_dim]
        batch_size = embeddings_sentence.shape[0]
        bos_embedding = self.model.get_input_embeddings().weight[self.tokenizer.bos_token_id].unsqueeze(0).unsqueeze(0) # [1, 1, embedding_dim]
        bos_embedding = bos_embedding.repeat(batch_size, 1, 1) # [batch_size, 1, embedding_dim]
        modified_embeddings_sentence = torch.cat((bos_embedding, embeddings_sentence[:, :-1, :]), dim=1) # [batch_size, seq_len, embedding_dim]
        outputs = self.model(inputs_embeds=modified_embeddings_sentence)
        probabilities = torch.softmax(outputs.logits, dim=-1) # [batch_size, seq_len, vocab_size]
        embedding_matrix = self.model.get_input_embeddings().weight # [vocab_size, embedding_dim]
        embedding_matrix = embedding_matrix.repeat(batch_size, 1, 1) # [batch_size, vocab_size, embedding_dim]
        WEx = torch.bmm(embedding_matrix, embeddings_sentence.transpose(1, 2)) # [batch_size, vocab_size, seq_len]
        
        # indices of the closest embeddings in the embedding matrix
        indices = torch.argmax(WEx, dim=1) # [batch_size, seq_len]

        # gather along the last dimension (vocab_size)
        cond_prob = torch.gather(input=probabilities, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1) # [batch_size, seq_len]
        
        log_likelihood = torch.sum(torch.log(cond_prob), dim=-1)
        return log_likelihood # [batch_size]

    def fit(self, target, nb_epochs, optimizer, alpha=1, beta=1, gamma=1):
        self.peft_model.train()
        pbar = tqdm(range(nb_epochs))
        for epoch in pbar:
            optimizer.zero_grad()

            # original loss
            target_tok = self.tokenizer(target, return_tensors='pt')
            target_ids = target_tok['input_ids'].to(self.device)
            attention_mask = target_tok['attention_mask'].to(self.device)
            outputs = self.peft_model(input_ids=target_ids, attention_mask=attention_mask, labels=target_ids)
            loss = outputs.loss

            prompt_embeddings = self.peft_model.get_prompt(batch_size=1)
            entropy = self.compute_entropy(prompt_embeddings)
            likelihood_loss = -self.log_likelihood(prompt_embeddings)[0]

            total_loss = alpha*loss + beta*entropy + gamma*likelihood_loss

            total_loss.backward() # backward on the total loss
            optimizer.step()
        
        prompt_embeddings = self.peft_model.get_prompt(batch_size=1)
        embedding_matrix = self.peft_model.get_input_embeddings().weight.data
        nearest_tokens = torch.argmax(torch.matmul(prompt_embeddings, embedding_matrix.T), dim=-1) # Find the nearest tokens in the embedding space
        decoded_prompt = self.tokenizer.decode(nearest_tokens[0], skip_special_tokens=True)
        self.prompt = decoded_prompt
    
    def generate_from_embeddings(self, max_length=50):
        input_ids = self.tokenizer("", return_tensors="pt").to(self.device)
        output = self.peft_model.generate(**input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
