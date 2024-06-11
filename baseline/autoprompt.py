import torch
from torch import nn
from baseline.prompting import Prompting
from tqdm import tqdm
from einops import einsum

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
        self.prompt = "yes"*num_virtual_tokens # "yes" corresponds to one token
    
    def forward_and_backward(self, batch):
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        out.loss.backward()
        grad = self.grad_storage.get_grad()
        with torch.no_grad():
            gradient_dot_embeddings = einsum(
                self.embeddings.weight, grad, "v h, b l h -> b v l"
            )
        return out, gradient_dot_embeddings

    def fit(self, target, nb_epochs, k, approximation=False):
        self.model.train()

        # prompt tokenization
        batch = self.tokenizer(
            [self.prompt+target],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # batch
        batch["labels"] = batch["input_ids"].clone()
        batch["decoder_attention_mask"] = batch["attention_mask"]
        batch = {k: v.to(self.device) for k, v in batch.items()}

        pbar = tqdm(range(nb_epochs))
        for epoch in pbar:
            out, gradient_dot_embeddings = self.forward_and_backward(batch)
            top_tokens_candidates = torch.topk(-gradient_dot_embeddings, k, dim=1).indices

            for i in range(self.num_virtual_tokens):
                if approximation:
                    best_token = top_tokens_candidates[0, 0, i].item() # le premier car meilleure approximation
                else: # run the topk tokens and choose the one with the minimal loss
                    best_token = top_tokens_candidates[0, 0, i].item() # premier token du premier batch
                    input_ids = batch["input_ids"].clone()
                    input_ids[0,i] = best_token
                    best_loss = self.model(
                                    input_ids=input_ids,
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"],
                                ).loss
                    for candidate in top_tokens_candidates[0,:,i]:
                        input_ids = batch["input_ids"].clone()
                        input_ids[0,i] = candidate.item()
                        out = self.model(
                            input_ids=input_ids,
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        if out.loss < best_loss:
                            best_token = candidate.unsqueeze(dim=0)
                            best_loss = out.loss

                # update
                batch["input_ids"][0][i] = best_token

        # prompt in a single string
        tokens = self.tokenizer.convert_ids_to_tokens(batch["input_ids"][0][:self.num_virtual_tokens])
        self.prompt = self.tokenizer.convert_tokens_to_string(tokens)
