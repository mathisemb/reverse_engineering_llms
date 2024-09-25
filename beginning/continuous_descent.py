from functools import partial

import torch
from einops import einsum
from torch import func, nn
from torch.func import jacrev
from tqdm import tqdm
import torch.nn.functional as F
#import matplotlib.pyplot as plt


class GradientStorage:
    def __init__(self, module: nn.Module):
        self.stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self.stored_gradient = grad_out[0]

    def get_grad(self):
        return self.stored_gradient


class GradExtractorEncoderDecoder:
    def __init__(self, model: nn.Module, embeddings: nn.Embedding):
        self.model = model
        self.embeddings = embeddings
        self.grad_storage = GradientStorage(self.embeddings)

    def highest_logprobs(self, inputs_embeds, attention_mask, labels):
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = out.logits
        logprobs = torch.log_softmax(logits, dim=-1)
        max_logprobs = torch.max(logprobs, dim=-1)

        return max_logprobs.values[0, -1]

    def topk_logprobs(self, inputs_embeds, attention_mask, labels, topk=10):
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = out.logits
        logprobs = torch.log_softmax(logits, dim=-1)
        topk = min(topk, logprobs.size(-1))
        # (B L K)
        topk_out = torch.topk(logprobs, topk, dim=-1, sorted=True)
        return topk_out.values.sum(0)

    def grad_wrt_next_logprobs(self, batch):
        grad = func.grad(self.highest_logprobs)(
            self.embeddings(batch["input_ids"]),
            batch["attention_mask"],
            batch["labels"],
        )
        return grad

    def jacobian_wrt_next_topk_logprobs(self, batch, topk=10):
        inputs_embeds = self.embeddings(batch["input_ids"])
        topk_func = partial(
            self.topk_logprobs,
            batch["attention_mask"],
            batch["labels"],
            topk,
        )
        compute_batch_jacobian = jacrev(topk_func)
        jacobian = compute_batch_jacobian(inputs_embeds)

        return jacobian

    def forward_and_backward(self, batch):
        print("==========input_ids==========")
        print(batch["input_ids"])
        print("==========labels==========")
        print(batch["labels"])
        out = self.model(
            inputs_embeds=batch["inputs_embeds"],
            #input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        out.loss.backward(retain_graph=True) # retain_graph=True to perform subsequent backward passes or access intermediate tensors as needed
        grad = self.grad_storage.get_grad()
        with torch.no_grad():
            gradient_dot_embeddings = einsum(
                self.embeddings.weight, grad, "v h, b l h -> b v l"
            )
        return out, grad, gradient_dot_embeddings
    
    def closest_word(self, embedding):
        """
        # cosine similarity
        similarities = F.cosine_similarity(embedding.unsqueeze(0), self.embeddings.weight, dim=1)
        closest_id = torch.argmax(similarities).item()
        closest_emb = self.embeddings.weight[closest_id]
        """
        # dot product
        dot_products = torch.einsum('i,ji->j', embedding, self.embeddings.weight)
        closest_id = torch.argmax(dot_products).item()
        closest_emb = self.embeddings.weight[closest_id]

        return closest_emb, closest_id

    def continuous_descent(self, input_str, label_str, suffix_length, nb_steps, lr):
        """
        computes the optimal tokens to write as a suffix of the user input to generate a given label (token that we want the model to predict)

        input_str: user input we should not modify
        label_str: token we aim to generate
        """
        input_length = len(tokenizer([input_str],padding=True,truncation=True,max_length=512,return_tensors="pt",)["input_ids"][0])-1

        # input tokenization
        batch = tokenizer.encode(
            input_str,
            padding=False,
            truncation=True,
            max_length=512,
            add_special_tokens = False
        )
        prompt_ids = torch.tensor([5]*suffix_length + batch + [tokenizer.eos_token_id]).unsqueeze(0) # on rajoute suffix_length tokens '.'
        prompt_ids = prompt_ids.to(DEVICE)
        print("batch", batch)
        batch = {"input_ids":prompt_ids,"attention_mask":torch.ones_like(prompt_ids, device=DEVICE)}
        batch["inputs_embeds"] = self.embeddings(prompt_ids).clone() # self.embeddings.weight[prompt_ids] empeche le hook de prendre tous les embeddings
        prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids[0])
        print("Initial prompt tokens:", prompt_tokens)

        # label tokenization
        labels_encoding = tokenizer(
            [
                label_str,
            ],
            add_special_tokens=False,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        labels_ids = labels_encoding["input_ids"]
        print("labels tokens:", tokenizer.convert_ids_to_tokens(labels_ids[0]))
        batch["labels"] = labels_encoding["input_ids"]

        batch["decoder_attention_mask"] = labels_encoding["attention_mask"]
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        gradient_norms = []

        for step in tqdm(range(nb_steps)):
            out, grad, gradient_dot_embeddings = self.forward_and_backward(batch)

            gradient_norms.append(torch.norm(grad, p=2))

            """
            for i in range(suffix_length):
                # grad.shape = [batch, prompt_length, embedding_size] = [1, 19, 768]
                batch["inputs_embeds"][0, i] -= lr * grad[0, i] / grad[0, i].norm(dim=-1, keepdim=True)
            """
            batch["inputs_embeds"][:,:suffix_length,:] -= lr * grad[:,:suffix_length,:] / grad[:,:suffix_length,:].norm(dim=-1, keepdim=True)

        """
        embeds = batch["inputs_embeds"].clone()
        batch_return = batch
        batch_return["inputs_embeds"] = embeds
        """

        for i in range(suffix_length):
            closest_emb, closest_id = self.closest_word(batch["inputs_embeds"][0][i])
            #batch["inputs_embeds"][0][i] = closest_emb
            prompt_ids[0][i] = closest_id

        #return batch["input_ids"], tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
        return prompt_ids, tokenizer.convert_ids_to_tokens(prompt_ids[0]), batch, gradient_norms


if __name__ == "__main__":
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.resize_token_embeddings(len(tokenizer))

    input_str = "London is the capital of <extra_id_0>"
    label_str = "<extra_id_0> France"

    grad_extractor = GradExtractorEncoderDecoder(
        model=model, embeddings=model.encoder.embed_tokens
    )
    # the forward function automatically creates the correct decoder_input_ids
    model.to(DEVICE)

    suffix_length = 10
    nb_steps = 100
    lr = 10
    opt_input_ids, opt_input_tokens, batch, gradient_norms = grad_extractor.continuous_descent(input_str, label_str, suffix_length, nb_steps, lr)
    print("Optimized prompt:", opt_input_tokens)

    #plt.plot(gradient_norms)
    #plt.show()

    # Generation test
    print("\n--------Generated with the user input--------")
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids
    outputs = model.generate(input_ids.to(DEVICE))
    print("model(user input):", tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("\n--------Generated with the optimized prompt--------")
    new_outputs = model.generate(opt_input_ids.to(DEVICE))
    print("model(closest tokens from opt embeddings):", tokenizer.decode(new_outputs[0], skip_special_tokens=True))

    # --- with embeddings as inputs ---
    #import ipdb
    #ipdb.set_trace()
    out = model(
        inputs_embeds=batch["inputs_embeds"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    #print("out.logits", out.logits.shape)
    argmax_output_token = torch.tensor([torch.argmax(out.logits[0,1]).item()])
    print("model(opt embeddings):", tokenizer.decode(argmax_output_token, skip_special_tokens=True))

    # --- with embeddings as inputs and CHANGING THE USER INPUT ---
    out = model(
        inputs_embeds=batch["inputs_embeds"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    #print("out.logits", out.logits.shape)
    argmax_output_token = torch.tensor([torch.argmax(out.logits[0,1]).item()])
    print("model(opt embeddings):", tokenizer.decode(argmax_output_token, skip_special_tokens=True))
