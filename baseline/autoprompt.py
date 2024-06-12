import torch
from torch import nn
from baseline.prompting import Prompting
from tqdm import tqdm
from einops import einsum
import torch.nn.functional as F

class GradientStorage:
    def __init__(self, module: nn.Module):
        self.stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self.stored_gradient = grad_out[0]

    def get_grad(self):
        return self.stored_gradient
    
class AutoPrompt(Prompting):
    def __init__(self, model_name, device, num_virtual_tokens):
        super().__init__(model_name, device)
        self.embeddings = self.model.get_input_embeddings()
        self.grad_storage = GradientStorage(self.embeddings)
        self.num_virtual_tokens = num_virtual_tokens
        self.prompt = "yes"*num_virtual_tokens # ("yes" corresponds to one token)
        self.prompt_ids = self.tokenizer(self.prompt, return_tensors='pt')['input_ids'][0]
    
    def update_prompt(self):
        self.prompt = self.tokenizer.decode(self.prompt_ids.tolist())
    
    def compute_loss(self, target):
        # prepare inputs
        target_ids = self.tokenizer(target, return_tensors='pt')['input_ids'].to(self.device)
        inputs_tok = self.tokenizer(self.prompt + target, return_tensors='pt')
        inputs_ids = inputs_tok['input_ids'].to(self.device)
        attention_mask = inputs_tok['attention_mask'].to(self.device)
        target_length = target_ids.shape[1]
        
        # run p on prompt + target
        out = self.model(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            labels=inputs_ids,
        )

        # computes -log(p(target|prompt))
        distributions = F.softmax(out.logits, dim=-1) # [batch_size, input_len, vocab_size]
        distributions = distributions[0] # [input_len, vocab_size]
        selected_distributions = distributions[-(target_length+1):-1, :] # [target_len, vocab_size]
        # for each token in the prompt, we want to get the probability of the corresponding token in the target
        probabilities = torch.zeros(target_length).to(self.device)
        for i in range(target_length):
            probabilities[i] = selected_distributions[i, target_ids[0,i]] # we take the probability of generating the i-th token of the target
        loss = -torch.sum(torch.log(probabilities), dim=-1)

        return loss
    
    def forward_and_backward(self, target):
        loss = self.compute_loss(target) # forward pass
        loss.backward() # backward pass
        grad = self.grad_storage.get_grad()
        with torch.no_grad():
            gradient_dot_embeddings = einsum(
                self.embeddings.weight, grad, "v h, b l h -> b v l"
            )
        return gradient_dot_embeddings

    def fit(self, target, nb_epochs, k, approximation=False):
        self.model.train()
        pbar = tqdm(range(nb_epochs))
        for epoch in pbar:

            gradient_dot_embeddings = self.forward_and_backward(target)
            top_tokens_candidates = torch.topk(-gradient_dot_embeddings, k, dim=1).indices

            for i in range(self.num_virtual_tokens):
                if approximation:
                    best_token = top_tokens_candidates[0, 0, i].item() # le premier car meilleure approximation
                else: # run the topk tokens and choose the one with the minimal loss

                    # the i-th token of the prompt is replaced by the possible best token
                    best_token = top_tokens_candidates[0, 0, i].item() # premier token du premier batch
                    self.prompt_ids[i] = best_token
                    self.update_prompt()
                    with torch.no_grad():
                        best_loss = self.compute_loss(target).item() # best loss corresponding to the current best token

                    # see if there is a better token
                    for candidate in top_tokens_candidates[0,:,i]:
                        self.prompt_ids[i] = candidate
                        self.update_prompt()
                        with torch.no_grad():
                            candidate_loss = self.compute_loss(target).item() # best loss corresponding to the current best token
                        if candidate_loss < best_loss:
                            best_token = candidate.item()
                            best_loss = candidate_loss

                # update
                self.prompt_ids[i] = best_token
                self.update_prompt()
