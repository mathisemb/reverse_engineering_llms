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

    def fit(self, target, nb_epochs, optimizer, alpha=1, beta=1, gamma=1, forward_proj=False):
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

            pbar.set_description(f"Loss: {loss:.4f}, Entropy: {entropy:.4f}, Likelihood: {likelihood_loss:.4f}")

        if forward_proj:
            # instead of doing pi = argmax_{e in V} <x_i, e>
            # we do pi = argmax_{e in V} <grad_xi 1/n sum_j=1...n D(dx_j|dx_j), e>
            # where dx_j is the distribution output for the j-th token
            # and grad_xi 1/n sum_j=1...n D(dx_j|dx_j) is the gradient
            # for x -> 1/n sum_j=1...n D(a_j|dx_j) where a_j = dx_j is fixed
            prompt_embeddings = self.peft_model.get_prompt(batch_size=1) # [1, num_virtual_tokens, hidden_size]
            embedding_matrix = self.peft_model.get_input_embeddings().weight.data # [vocab_size, hidden_size]
            
            outputs = self.peft_model(input_ids=target_ids, attention_mask=attention_mask, labels=target_ids)
            fixed_distribs = F.softmax(outputs.logits, dim=-1) # [batch_size, seq_length, vocab_size]

            outputs = self.peft_model(input_ids=target_ids, attention_mask=attention_mask, labels=target_ids)
            variable_distribs = F.softmax(outputs.logits, dim=-1) # [batch_size, seq_length, vocab_size]

            # divergence for each virtual token, along the vocabulary
            divergences = torch.sum(fixed_distribs * torch.log(fixed_distribs / variable_distribs), dim=-1) # [batch_size, seq_length]
            mean_divergence = torch.mean(divergences, dim=-1) # [batch_size]

            # we now want the gradient of the mean divergence with respect to the trainable parameters in a tensor of shape [num_virtual_tokens, hidden_size]
            for param in self.peft_model.parameters():
                if param.requires_grad:
                    trainable_parameters = param
            print("trainable_parameters:", trainable_parameters.shape)
            gradients = torch.autograd.grad(mean_divergence, trainable_parameters)[0]
            print("gradients:", gradients.shape)
            
            # if gradient is of shape [num_virtual_tokens, hidden_size]
            nearest_tokens = torch.argmax(torch.matmul(gradients, embedding_matrix.T), dim=-1)
            decoded_prompt = self.tokenizer.decode(nearest_tokens, skip_special_tokens=True)
        else: # classic nearest token projection
            prompt_embeddings = self.peft_model.get_prompt(batch_size=1)
            embedding_matrix = self.peft_model.get_input_embeddings().weight.data
            nearest_tokens = torch.argmax(torch.matmul(prompt_embeddings, embedding_matrix.T), dim=-1) # Find the nearest tokens in the embedding space
            decoded_prompt = self.tokenizer.decode(nearest_tokens[0], skip_special_tokens=True)
        self.prompt = decoded_prompt
    
    def generate_from_embeddings(self, max_length=50):
        output = self.peft_model.generate(max_length=max_length) # no input, virtual embeddings are part of the model
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
