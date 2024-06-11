import torch
from baseline.prompting import Prompting
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from tqdm import tqdm

class NewMethod(Prompting):
    """
    for prompt tuning we use peft and

    self.model corresponds to the original model
    self.peft_model corresponds to the augmented peft model
    self.prompt corresponds to the virtual embeddings projected into the vocabulary
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
            
            loss = outputs.loss
            loss.backward() # backward on the total loss
            optimizer.step()
            optimizer.zero_grad()
        
        prompt_embeddings = self.peft_model.get_prompt(batch_size=1)
        embedding_matrix = self.peft_model.get_input_embeddings().weight.data
        nearest_tokens = torch.argmax(torch.matmul(prompt_embeddings, embedding_matrix.T), dim=-1) # Find the nearest tokens in the embedding space
        decoded_prompt = self.tokenizer.decode(nearest_tokens[0], skip_special_tokens=True)
        self.prompt = decoded_prompt
    
    def generate_from_embeddings(self, max_length=50):
        input_ids = self.tokenizer("", return_tensors="pt").to(self.device)
        output = self.peft_model.generate(**input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
