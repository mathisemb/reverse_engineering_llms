import torch
from baseline.prompting import Prompting
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from tqdm import tqdm
from torch.nn import functional as F

class ForwardProjection(Prompting):
    """
    for prompt tuning we use peft and

    self.model corresponds to the original model
    self.peft_model corresponds to the augmented peft model
    self.prompt corresponds to the virtual embeddings projected into the vocabulary
    (the virtual embeddings are accessible through self.peft_model.get_prompt()
    """

    def __init__(self, model_name, device, num_virtual_tokens):
        super().__init__(model_name, device)
        peft_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=num_virtual_tokens,
            tokenizer_name_or_path=model_name,
        )
        self.peft_model = get_peft_model(self.model, peft_config)

    def fit(self, target, nb_epochs, optimizer):
        self.peft_model.train()
        pbar = tqdm(range(nb_epochs))
        for epoch in pbar:
            optimizer.zero_grad()

            target_tok = self.tokenizer(target, return_tensors='pt')
            target_ids = target_tok['input_ids'].to(self.device)
            attention_mask = target_tok['attention_mask'].to(self.device)
            outputs = self.peft_model(input_ids=target_ids, attention_mask=attention_mask, labels=target_ids)
            
            loss = outputs.loss # there is no prefix, the virtual embeddings are part of the model hence we just take the loss
            loss.backward()
            optimizer.step()
        
        # Forward projection
        # instead of doing pi = argmax_{e in V} <x_i, e>
        # we do pi = argmax_{e in V} <grad_xi 1/n sum_j=1...n D(dx_j|dx_j), e>
        # where dx_j is the distribution output for the j-th token
        # and grad_xi 1/n sum_j=1...n D(dx_j|dx_j) is the gradient
        # for x -> 1/n sum_j=1...n D(a_j|dx_j) where a_j = dx_j is fixed
        prompt_embeddings = self.peft_model.get_prompt(batch_size=1) # [1, num_virtual_tokens, hidden_size]
        embedding_matrix = self.peft_model.get_input_embeddings().weight.data # [vocab_size, hidden_size]
        
        outputs = self.peft_model(input_ids=target_ids, attention_mask=attention_mask, labels=target_ids)
        fixed_distribs = F.softmax(outputs.logits, dim=-1) # [batch_size, seq_length, vocab_size]

        outputs = self.peft_model(input_ids=target_ids, attention_mask=attention_mask, labels=target_ids)
        variable_distribs = F.softmax(outputs.logits, dim=-1) # [batch_size, seq_length, vocab_size]

        # divergence for each virtual token, along the vocabulary
        divergences = torch.sum(fixed_distribs * torch.log(fixed_distribs / variable_distribs), dim=-1) # [batch_size, seq_length]
        mean_divergence = torch.mean(divergences, dim=-1) # [batch_size]

        # we now want the gradient of the mean divergence with respect to the trainable parameters in a tensor of shape [num_virtual_tokens, hidden_size]
        for param in self.peft_model.parameters():
            if param.requires_grad:
                trainable_parameters = param
        print("trainable_parameters:", trainable_parameters.shape)
        gradients = torch.autograd.grad(mean_divergence, trainable_parameters)[0]
        print("gradients:", gradients.shape)
        
        # if gradient is of shape [num_virtual_tokens, hidden_size]
        nearest_tokens = torch.argmax(torch.matmul(gradients, embedding_matrix.T), dim=-1)
        decoded_prompt = self.tokenizer.decode(nearest_tokens, skip_special_tokens=True)
        self.prompt = decoded_prompt
    
    def generate_from_embeddings(self, max_length=50):
        output = self.peft_model.generate(max_length=max_length) # no input, virtual embeddings are part of the model
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
