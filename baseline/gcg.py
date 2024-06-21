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

class GreedyCoordinateGradient(Prompting):
    def __init__(self, model_name, device, num_virtual_tokens):
        super().__init__(model_name, device)
        self.embeddings = self.model.get_input_embeddings()
        self.grad_storage = GradientStorage(self.embeddings)
        self.num_virtual_tokens = num_virtual_tokens
        self.prompt_ids = torch.tensor([0]*num_virtual_tokens).to(self.device)
        self.update_prompt()
    
    def update_prompt(self):
        self.prompt = self.tokenizer.decode(self.prompt_ids.tolist())

    def some_tests(self):
        # test compute_loss
        self.prompt = "The capital of France is"
        self.prompt_ids = self.tokenizer(self.prompt, add_special_tokens=False, return_tensors='pt')['input_ids'][0].to(self.device)
        print("self.prompt", self.prompt, "\n")

        loss = self.compute_loss("Paris")
        print("self.compute_loss(Paris) = ", loss)
        out = self.model(self.prompt_ids.unsqueeze(0))
        paris_token_id = self.tokenizer("Paris", add_special_tokens=False, return_tensors='pt')['input_ids'][0].to(self.device)[0]
        rank = torch.argsort(out.logits[0, -1, :], descending=True).tolist().index(paris_token_id)
        print("rank of France:", rank, "/", out.logits.shape[-1], "\n")

        loss = self.compute_loss("London")
        print("self.compute_loss(London) = ", loss)
        out = self.model(self.prompt_ids.unsqueeze(0))
        london_token_id = self.tokenizer("London", add_special_tokens=False, return_tensors='pt')['input_ids'][0].to(self.device)[0]
        rank = torch.argsort(out.logits[0, -1, :], descending=True).tolist().index(london_token_id)
        print("rank of London:", rank, "/", out.logits.shape[-1], "\n")
    
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
    
    def compute_loss_2(self, prompt_ids, target):
        target_ids = self.tokenizer(target, add_special_tokens=False, return_tensors='pt')['input_ids'][0].to(self.device)
        target_length = len(target_ids)
        inputs_ids = torch.cat((prompt_ids, target_ids), dim=0).unsqueeze(0)
        out = self.model(inputs_ids)
        distributions = F.softmax(out.logits, dim=-1)[0]  # [input_len, vocab_size], remove batch dimension
        selected_distributions = distributions[-(target_length+1):-1, :]  # [target_len, vocab_size]
        target_probs = selected_distributions[torch.arange(target_length), target_ids]
        return -torch.sum(torch.log(target_probs))
    
    def forward_and_backward(self, target):
        loss = self.compute_loss(target) # forward pass
        loss.backward() # backward pass
        grad = self.grad_storage.get_grad() # [batch_size, num_virtual_tokens+target_length, hidden_size]
        with torch.no_grad():
            gradient_dot_embeddings = einsum(
                self.embeddings.weight, grad, "v h, b l h -> b v l"
            ) # [batch_size, vocab_size, num_virtual_tokens+target_length]
        return loss, gradient_dot_embeddings

    def fit(self, target, nb_epochs=20, k=5, batch_size=10):
        self.model.train()
        training_losses = []
        ranks = []
        pbar = tqdm(range(nb_epochs))
        for epoch in pbar:

            loss, gradient_dot_embeddings = self.forward_and_backward(target) # [batch_size, vocab_size, num_virtual_tokens+target_length]
            top_tokens_candidates = torch.topk(-gradient_dot_embeddings, k, dim=1).indices # [batch_size, k, num_virtual_tokens+target_length]
            training_losses.append(loss.item())

            prompts = self.prompt_ids.clone().repeat(batch_size, 1)
            for b in range(batch_size):
                prompts[b] = self.prompt_ids.clone()
                i = torch.randint(0, self.num_virtual_tokens, (1,)).item()
                # i = Unif(0, num_virtual_tokens-1)
                # prompts[b,i] = Unif(kcandidates_i)
                prompts[b,i] = top_tokens_candidates[0, torch.randint(0, k, (1,)).item(), i].item()
            
            min_loss = float('inf')
            for b in range(batch_size):
                loss = self.compute_loss_2(prompts[b], target).item()
                if loss < min_loss:
                    min_loss = loss
                    best_prompt = prompts[b]
            
            self.prompt_ids = best_prompt
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
                print("5 most probable tokens:", self.tokenizer.decode(torch.topk(out.logits[0, -1, :], 5).indices))
                distrib = F.softmax(out.logits[0, -1, :], dim=-1)
                entropy = -torch.sum(distrib * torch.log(distrib)).item()
                print("Entropy of the next token distribution:", entropy, "\n")

            pbar.set_description(f"Loss: {loss:.4f}")

        max_rank = max(ranks)
        ranks = [rank / max_rank for rank in ranks]

        return training_losses, ranks
