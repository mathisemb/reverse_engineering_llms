import torch
from baseline.prompting import Prompting
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from tqdm import tqdm

class PromptTuning(Prompting):
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

    def fit(self, target, nb_epochs=20):
        self.peft_model.train()
        optimizer = torch.optim.Adam(self.peft_model.parameters(), lr=3e-2)
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

            with torch.no_grad():
                # forward without input
                outputs = self.peft_model(input_ids=self.tokenizer("", return_tensors='pt')['input_ids'].to(self.device))
                print("The 5 most probable tokens:", self.tokenizer.decode(torch.topk(outputs.logits[0, -1], k=5).indices, skip_special_tokens=True))
                print("With probabilities:", torch.topk(torch.softmax(outputs.logits[0, -1], dim=-1), k=5).values)
        
        # Find the nearest tokens in the embedding space
        prompt_embeddings = self.peft_model.get_prompt(batch_size=1)
        embedding_matrix = self.peft_model.get_input_embeddings().weight.data
        nearest_tokens = torch.argmax(torch.matmul(prompt_embeddings, embedding_matrix.T), dim=-1)
        decoded_prompt = self.tokenizer.decode(nearest_tokens[0], skip_special_tokens=True)
        self.prompt = decoded_prompt
    
    def generate_from_embeddings(self, max_length=50):
        output = self.peft_model.generate(max_length=max_length) # no input, virtual embeddings are part of the model
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
