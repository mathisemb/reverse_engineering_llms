from functools import partial

import torch
from einops import einsum
from torch import func, nn
from torch.func import jacrev


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
        with torch.no_grad():
            gradient_dot_embeddings = einsum(
                self.embeddings.weight, grad, "v h, b l h -> b v l"
            )
        return out, gradient_dot_embeddings

    def gradient_descent(self, input_str, label_str):
        """
        computes the optimal tokens to write as a suffix of the user input to generate a given label (token that we want the model to predict)

        input_str: user input we should not modify
        label_str: token we aim to generate
        """
        prompt_str = input_str + "<extra_id_2><extra_id_3><extra_id_4><extra_id_5><extra_id_6><extra_id_7><extra_id_8><extra_id_9><extra_id_10><extra_id_11>"

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
        input_tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
        print("input:", input_tokens)

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
        #print("batch:", batch)

        nb_steps = 1000
        for step in range(nb_steps):
            out, gradient_dot_embeddings = self.forward_and_backward(batch)
            
            top_tokens_values = torch.topk(-gradient_dot_embeddings, 5, dim=1).values
            top_tokens_candidates = torch.topk(-gradient_dot_embeddings, 5, dim=1).indices
            #for i in range(gradient_dot_embeddings.shape[-1]):
            for i in range(8,18):
                best_token = top_tokens_candidates[0, :, i][0].item()
                best_word = tokenizer.decode(best_token, skip_special_tokens=True)

                #print("We want it replace:", input_tokens[i])
                #print("best token:", best_token)
                #print("best word:", best_word)

                # update
                batch["input_ids"][0][i] = best_token

            if step%100==0:
                print("prompt:", tokenizer.convert_ids_to_tokens(batch["input_ids"][0]))

        #print("batch[input_ids]:", batch["input_ids"])
        return batch["input_ids"]



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

    new_input_ids = grad_extractor.gradient_descent(input_str, label_str)


    # Generation test
    print("\n--------Generate with input_ids--------")
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids
    print("input_ids:", input_ids)
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("\n--------Generate with new_input_ids--------")
    print("new_input_ids:", new_input_ids)
    new_outputs = model.generate(new_input_ids)
    print(tokenizer.decode(new_outputs[0], skip_special_tokens=True))
