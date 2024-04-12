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
        inputs_embeds = self.embeddings(batch["input_ids"])
        grad = func.grad(self.highest_logprobs)(
            inputs_embeds,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return grad

    def jacobian_wrt_next_topk_logprobs(self, batch, topk=10):
        inputs_embeds = self.embeddings(batch["input_ids"])
        topk_func = partial(
            self.topk_logprobs,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            topk=topk,
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


if __name__ == "__main__":
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    input_str = "London is the capital of <extra_id_0>"
    label_str = "<extra_id_0> France <extra_id_2>"

    # Create input ids with some masks.
    batch = tokenizer(
        [
            input_str,
        ],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
    print(input_tokens)

    # What the model should guess.
    # The answer corresponding to the input mask <extra_id_0> is "cute dog", the answer to <extra_id_1> is "the".
    # Label corresponding to <extra_id_i> in the input id is surrounded by <extra_id_i> "ANSWER" <extra_id_i+1> in the label.
    labels_encoding = tokenizer(
        [
            label_str,
        ],
        add_special_tokens=False,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    batch["labels"] = labels_encoding["input_ids"]
    labels_non_pad_token = labels_encoding["attention_mask"]
    batch["decoder_attention_mask"] = labels_encoding["attention_mask"]

    print(tokenizer.convert_ids_to_tokens(batch["labels"][0]))

    grad_extractor = GradExtractorEncoderDecoder(
        model=model, embeddings=model.encoder.embed_tokens
    )
    # the forward function automatically creates the correct decoder_input_ids
    model.to(DEVICE)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    out, gradient_dot_embeddings = grad_extractor.forward_and_backward(batch)
    top_tokens_candidates = torch.topk(-gradient_dot_embeddings, 5, dim=1).indices
    for i in range(gradient_dot_embeddings.shape[-1]):
        print(
            f"Top 5 replacements for {input_tokens[i]}: ",
            tokenizer.decode(
                top_tokens_candidates[0, :, i].tolist(), skip_special_tokens=True
            ),
        )
