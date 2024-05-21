from functools import partial

import torch
from einops import einsum
from torch import func, nn
from torch.func import jacrev
from tqdm import tqdm
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
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        out.loss.backward()
        grad = self.grad_storage.get_grad()
        #print("grad", grad)
        #print("grad normed ", grad / grad.norm(dim=-1, keepdim=True))
        #grad = grad / grad.norm(dim=-1, keepdim=True)
        #print("grad shape", grad.shape)
        with torch.no_grad():
            gradient_dot_embeddings = einsum(
                self.embeddings.weight, grad, "v h, b l h -> b v l"
            )
        return out, gradient_dot_embeddings, grad

    def discrete_descent(self, input_str, label_str, suffix_length, nb_steps, k=5, approximation=False):
        """
        computes the optimal tokens to write as a suffix of the user input to generate a given label (token that we want the model to predict)

        input_str: user input we should not modify
        label_str: token we aim to generate
        """
        input_length = len(tokenizer([input_str],padding=True,truncation=True,max_length=512,return_tensors="pt",)["input_ids"][0])-1
        print("input_length:", input_length)

        prompt_str = ""
        for i in range(suffix_length):
            prompt_str += "<extra_id_" + str(i) + ">"
        prompt_str += input_str

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
        prompt_tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
        print("input:", prompt_tokens)

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
        print("labels:", tokenizer.convert_ids_to_tokens(labels_encoding["input_ids"][0]))

        # batch
        batch["labels"] = labels_encoding["input_ids"]
        batch["decoder_attention_mask"] = labels_encoding["attention_mask"]
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        gradient_norms = []

        for step in tqdm(range(nb_steps)):
            out, gradient_dot_embeddings, grad = self.forward_and_backward(batch)

            gradient_norms.append(torch.norm(grad, p=2))
            
            top_tokens_candidates = torch.topk(-gradient_dot_embeddings, k, dim=1).indices

            #for i in range(input_length,input_length+suffix_length):
            for i in range(suffix_length):

                if approximation:
                    best_token = top_tokens_candidates[0, 0, i].item() # le premier car meilleure approximation
                else: # run the topk tokens and choose the one with the minimal loss
                    best_token = top_tokens_candidates[0, 0, i].item() # premier token du premier batch
                    input_ids = batch["input_ids"].clone()
                    input_ids[0,i] = best_token
                    best_loss = model(
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
                
                best_word = tokenizer.decode(best_token, skip_special_tokens=True)

                """
                print("We want it replace:", prompt_tokens[i])
                print("best token:", best_token, "which corresponds to", best_word)
                """

                # update
                batch["input_ids"][0][i] = best_token

            """
            if step%100==0:
                print("prompt:", tokenizer.convert_ids_to_tokens(batch["input_ids"][0]))
            """

        #print("batch[input_ids]:", batch["input_ids"])
        return batch["input_ids"], tokenizer.convert_ids_to_tokens(batch["input_ids"][0]), gradient_norms


if __name__ == "__main__":
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    """
    input_str = "London is the capital of <extra_id_0>."
    label_str = "<extra_id_0> France <extra_id_1>"
    
    input_str = "United <extra_id_0>"
    label_str = "<extra_id_0> Kingdom"
    """
    input_str = "London is the capital of <extra_id_0>"
    label_str = "<extra_id_0> France"

    grad_extractor = GradExtractorEncoderDecoder(
        model=model, embeddings=model.encoder.embed_tokens
    )
    # the forward function automatically creates the correct decoder_input_ids
    model.to(DEVICE)

    suffix_length = 20
    nb_steps = 100
    k = 3
    opt_input_ids, opt_input_tokens, gradient_norms = grad_extractor.discrete_descent(input_str, label_str, suffix_length, nb_steps, k, approximation=False)
    print("Optimized prompt:", opt_input_tokens)

    #plt.plot(gradient_norms)
    #plt.show()

    # Generation test
    print("\n--------Generated with the user input--------")
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids
    outputs = model.generate(input_ids.to(DEVICE))
    print("model(opt embeddings):", tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("\n--------Generated with the optimized prompt--------")
    new_outputs = model.generate(opt_input_ids.to(DEVICE))
    print("model(opt tokens):", tokenizer.decode(new_outputs[0], skip_special_tokens=True))

    # Test with other cities
    """
    print("opt_input_ids", opt_input_ids)
    print("opt_input_ids", opt_input_ids[0,suffix_length].item())
    madrid = tokenizer(
        [
            "Madrid",
        ],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    madrid_id = madrid['input_ids'][0,0].item()
    print("madrid:", madrid_id)
    print("madrid?", tokenizer.convert_ids_to_tokens(madrid_id))
    """

    cities = ["Madrid", "Luxembourg", "Berlin", "Bruxelle", "Amsterdam", "Lisbonne"]
    for city in cities:
        city_id = tokenizer(
                    [
                        city,
                    ],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )['input_ids'][0,0].item()
        opt_input_ids[0,suffix_length] = city_id
        new_outputs = model.generate(opt_input_ids.to(DEVICE))
        print("model(opt tokens + " + city + "):", tokenizer.decode(new_outputs[0], skip_special_tokens=True))
