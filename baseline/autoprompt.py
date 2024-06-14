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
        self.prompt_ids = torch.tensor([0]*num_virtual_tokens)
        self.update_prompt()
    
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

        # Compute -log(p(target|prompt)) from p(prompt + target)
        distributions = F.softmax(out.logits, dim=-1)[0]  # [input_len, vocab_size], remove batch dimension
        selected_distributions = distributions[-(target_length+1):-1, :]  # [target_len, vocab_size]
        # Gather the probabilities for the target tokens
        # dimension 1: vocab_size
        # index: target_ids[0, :target_length].unsqueeze(1) -> [target_length, 1]
        # for every element of the last target_length tokens of selected_distributions, we pick the element (proba) corresponding to the target token ids
        target_probs = selected_distributions.gather(1, target_ids[0, :target_length].unsqueeze(1)).squeeze()
        # Compute the negative log likelihood loss
        loss = -torch.sum(torch.log(target_probs), dim=-1)

        return loss
    
    def forward_and_backward(self, target):
        loss = self.compute_loss(target) # forward pass
        loss.backward() # backward pass
        grad = self.grad_storage.get_grad()
        with torch.no_grad():
            gradient_dot_embeddings = einsum(
                self.embeddings.weight, grad, "v h, b l h -> b v l"
            )
        return loss, gradient_dot_embeddings

    def fit(self, target, nb_epochs, k, approximation=False):
        self.model.train()
        pbar = tqdm(range(nb_epochs))
        for epoch in pbar:

            loss, gradient_dot_embeddings = self.forward_and_backward(target) # [batch_size, vocab_size, num_virtual_tokens+target_length]
            top_tokens_candidates = torch.topk(-gradient_dot_embeddings, k, dim=1).indices # [batch_size, k, num_virtual_tokens+target_length]

            for i in range(self.num_virtual_tokens): # for each virtual token we select the one that minimizes the loss
                if approximation:
                    best_token = top_tokens_candidates[0, 0, i].item() # le premier car meilleure approximation
                else: # run the model with the topk tokens and choose the one with the minimal loss

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
            
            pbar.set_description(f"Loss: {loss:.4f}, Prompt: {self.prompt}")
