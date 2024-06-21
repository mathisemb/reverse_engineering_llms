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
    def fit(self, target, nb_epochs):
        pass
    
    def generate(self, max_length=100):
        input_ids = self.tokenizer.encode(self.prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
