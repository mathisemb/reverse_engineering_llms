from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

class Prompting(ABC):
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        self.prompt = "" # virtual tokens to be added to the input

    @abstractmethod
    def get_prompt(self):
        pass

    @abstractmethod
    def fit(self, target, nb_epochs):
        pass

    def get_prompt(self):
        return self.prompt
    
    def evaluate(self, dataset, target, batch_size=8, max_length=50):
        # run the model with the optimized prompt and return the score
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        score = 0
        for batch in dataloader:
            batch_length = len(batch)
            outputs = self.generate(batch, max_length)
            targets = target.repeat(batch_length, 1)
            score += sum([targets[i] in outputs[i] for i in range(batch_length)])
        score = score / ((len(dataloader)-1)*batch_size + batch_length)
        return score

    def generate(self, max_length=50):
        input_ids = self.tokenizer.encode(self.prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
