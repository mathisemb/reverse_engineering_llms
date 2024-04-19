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
        self.grad_storage = GradientStorage(self.embeddings) # we want to store gradient wrt to the embeddings

    def highest_logprobs(self, inputs_embeds, attention_mask, labels):
        out = self.model( # training call so it computes the gradients
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = out.logits
        logprobs = torch.log_softmax(logits, dim=-1)
        max_logprobs = torch.max(logprobs, dim=-1)

        return max_logprobs.values[0, -1]

    def topk_logprobs(self, inputs_embeds, attention_mask, labels, topk=10):
        out = self.model( # training call so it computes the gradients
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

    def grad_wrt_next_logprobs(self, batch): # (is not used yet)
        # returns grad_input highest_logprobs(batch)
        grad = func.grad(self.highest_logprobs)( # highest_logprobs : (input, mask, label) -> token, and we want to derive it
            inputs_embeds=self.embeddings(batch["input_ids"]), # input of highest_logprobs
            attention_mask=batch["attention_mask"], # input of highest_logprobs
            labels=batch["labels"], # input of highest_logprobs
        )
        return grad

    def jacobian_wrt_next_topk_logprobs(self, batch, topk=10): # (is not used yet)
        # returns J_model(batch)
        inputs_embeds = self.embeddings(batch["input_ids"])
        topk_func = partial(
            self.topk_logprobs, # # topk_logprobs : (input, mask, label) -> token^k, and we want to derive it
            attention_mask=batch["attention_mask"], # input
            labels=batch["labels"], # input
            topk=topk, # input
        ) # gives a new function with partial application the inputs: attention_mask, labels and topk
        compute_batch_jacobian = jacrev(topk_func) # function that returns the Jacobian of topk_func
        jacobian = compute_batch_jacobian(inputs_embeds)

        return jacobian

    def forward_and_backward(self, batch):
        out = self.model( # training call so it computes the gradients
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"], # input when training call so it can compute the loss
        )
        # https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/t5/modeling_t5.py#L1779
        out.loss.backward() # compute gradient of the loss (hence comparing output with labels)
        grad = self.grad_storage.get_grad() # get the gradient wrt to all the embeddings of the vocab
        print("grad:", grad.shape)
        print("embd:", self.embeddings.weight.shape)
        with torch.no_grad():
            gradient_dot_embeddings = einsum(
                self.embeddings.weight, grad, "v h, b l h -> b v l"
            ) # forall example of the batch, forall word w in the vocab, forall word x in the sentence, computes w^T.grad_emb Loss(x)
        return out, gradient_dot_embeddings # how much each word embedding contributes to the gradient at each position in the sequence (for each example in the batch)


if __name__ == "__main__":
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    input_str = "London is the capital of <extra_id_0>."
    label_str = "<extra_id_0> France <extra_id_1>"

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
    print("input:", input_tokens)

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

    print("labels:", tokenizer.convert_ids_to_tokens(batch["labels"][0]))

    #print("model.encoder.embed_tokens:", model.encoder.embed_tokens)

    grad_extractor = GradExtractorEncoderDecoder(
        model=model, embeddings=model.encoder.embed_tokens
    )
    # the forward function automatically creates the correct decoder_input_ids
    model.to(DEVICE)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    out, gradient_dot_embeddings = grad_extractor.forward_and_backward(batch)
    #print("gradient_dot_embeddings:", gradient_dot_embeddings.shape)

    top_tokens_values = torch.topk(-gradient_dot_embeddings, 5, dim=1).values
    top_tokens_candidates = torch.topk(-gradient_dot_embeddings, 5, dim=1).indices
    for i in range(gradient_dot_embeddings.shape[-1]):
        print(
            f"Top 5 replacements for {input_tokens[i]}: ",
            tokenizer.decode(
                top_tokens_candidates[0, :, i].tolist(), skip_special_tokens=True
            ),
        )

    # Generation test
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
