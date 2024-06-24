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
        self.model = self.model.to(torch.bfloat16)
        self.embeddings = self.model.get_input_embeddings()
        self.grad_storage = GradientStorage(self.embeddings)
        self.num_virtual_tokens = num_virtual_tokens
        self.prompt_ids = torch.tensor([0]*num_virtual_tokens).to(self.device)
        self.update_prompt()
    
    def update_prompt(self):
        self.prompt = self.tokenizer.decode(self.prompt_ids.tolist())
    
    def compute_loss(self, target):
        # prepare inputs
        target_ids = self.tokenizer(target, add_special_tokens=False, return_tensors='pt')['input_ids'][0].to(self.device)
        target_length = len(target_ids)
        inputs_ids = torch.cat((self.prompt_ids, target_ids), dim=0).unsqueeze(0)

        # run p on prompt + target
        out = self.model(inputs_ids)

        # Compute -log(p(target|prompt)) from p(prompt + target)
        distributions = F.softmax(out.logits, dim=-1)[0]  # [input_len, vocab_size], remove batch dimension
        selected_distributions = distributions[-(target_length+1):-1, :]  # [target_len, vocab_size]
        
        target_probs = selected_distributions[torch.arange(target_length), target_ids]
        """ equivalent to
        target_probs = torch.zeros(target_length)
        for i in range(target_length):
            target_probs[i] = selected_distributions[i, target_ids[i]]
        """

        loss = -torch.sum(torch.log(target_probs))

        return loss
    
    def forward_and_backward(self, target):
        loss = self.compute_loss(target) # forward pass
        loss.backward() # backward pass
        grad = self.grad_storage.get_grad() # [batch_size, num_virtual_tokens+target_length, hidden_size]
        with torch.no_grad():
            gradient_dot_embeddings = einsum(
                self.embeddings.weight, grad, "v h, b l h -> b v l"
            ) # [batch_size, vocab_size, num_virtual_tokens+target_length]
        return loss, gradient_dot_embeddings

    def fit(self, target, nb_epochs=20, k=5, approximation=False):
        self.model.train()
        training_losses = []
        ranks = []
        pbar = tqdm(range(nb_epochs))
        for epoch in pbar:
            #import ipdb; ipdb.set_trace()

            loss, gradient_dot_embeddings = self.forward_and_backward(target) # [batch_size, vocab_size, num_virtual_tokens+target_length]
            top_tokens_candidates = torch.topk(-gradient_dot_embeddings, k, dim=1).indices # [batch_size, k, num_virtual_tokens+target_length]
            training_losses.append(loss.item())

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
            
            # some stats
            print("prompt:", self.prompt)
            with torch.no_grad():
                target_ids = self.tokenizer(target, add_special_tokens=False, return_tensors='pt')['input_ids'][0].to(self.device)
                inputs_ids = torch.cat((self.prompt_ids, target_ids), dim=0).unsqueeze(0)
                out = self.model(inputs_ids)
                first_token = target_ids[0]
                rank = torch.argsort(out.logits[0, -1, :], descending=True).tolist().index(first_token)
                print("rank of", self.tokenizer.decode(first_token), ":", rank, "/", out.logits.shape[-1], "\n")
                ranks.append(rank)

            pbar.set_description(f"Loss: {loss:.4f}")

        max_rank = max(ranks)
        ranks = [rank / max_rank for rank in ranks]

        return training_losses, ranks
