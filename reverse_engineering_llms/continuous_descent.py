from functools import partial

import torch
from einops import einsum
from torch import func, nn
from torch.func import jacrev
from tqdm import tqdm
import torch.nn.functional as F


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
        # returns the embedding of an existing token that is the closest from the input embedding (projection from embedding space to embedding space of real words)
        closest_id = 0
        closest_emb = self.embeddings.weight[closest_id]
        closest_sim = F.cosine_similarity(embedding, closest_emb, dim=0)
        for id, candidate in enumerate(self.embeddings.weight):
            sim = F.cosine_similarity(embedding, candidate, dim=0)
            if sim < closest_sim:
                closest_sim = sim
                closest_emb = candidate
                closest_id = id
        return closest_emb, closest_id

    def continuous_descent(self, input_str, label_str, suffix_length, nb_steps, lr):
        """
        computes the optimal tokens to write as a suffix of the user input to generate a given label (token that we want the model to predict)

        input_str: user input we should not modify
        label_str: token we aim to generate
        """
        input_length = len(tokenizer([input_str],padding=True,truncation=True,max_length=512,return_tensors="pt",)["input_ids"][0])-1

        prompt_str = input_str
        for i in range(suffix_length):
            prompt_str += "<extra_id_" + str(i) + ">"

        # input tokenization
        batch = tokenizer(
            [
                prompt_str,
            ],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        prompt_ids = batch["input_ids"]
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

        for step in tqdm(range(nb_steps)):
            out, grad, gradient_dot_embeddings = self.forward_and_backward(batch)

            for i in range(input_length,input_length+suffix_length):
                # grad.shape = [batch, prompt_length, embedding_size] = [1, 19, 768]
                batch["inputs_embeds"] -= lr * grad[0, i]

        for i in range(input_length,input_length+suffix_length):
            closest_emb, closest_id = self.closest_word(batch["inputs_embeds"][0][i])
            batch["inputs_embeds"][0][i] = closest_emb
            prompt_ids[0][i] = closest_id
            
        #return batch["input_ids"], tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
        return prompt_ids, tokenizer.convert_ids_to_tokens(prompt_ids[0])


if __name__ == "__main__":
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    input_str = "London is the capital of <extra_id_0>."
    label_str = "<extra_id_0> France <extra_id_1>"

    grad_extractor = GradExtractorEncoderDecoder(
        model=model, embeddings=model.encoder.embed_tokens
    )
    # the forward function automatically creates the correct decoder_input_ids
    model.to(DEVICE)

    suffix_length = 10
    nb_steps = 200
    lr = 0.1
    opt_input_ids, opt_input_tokens = grad_extractor.continuous_descent(input_str, label_str, suffix_length, nb_steps, lr)
    print("Optimized prompt:", opt_input_tokens)

    # Generation test
    print("\n--------Generated with the user input--------")
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("\n--------Generated with the optimized prompt--------")
    new_outputs = model.generate(opt_input_ids)
    print(tokenizer.decode(new_outputs[0], skip_special_tokens=True))
