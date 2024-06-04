from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoModelForCausalLM
from peft import PrefixTuningConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


# MODEL NAME
model_name = "bigscience/bloomz-560m"
#model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" #error with past_key_value


# LOAD DATASET
dataset = load_dataset("ought/raft", "twitter_complaints")
classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
# For example: dataset["train"][0] = {"Tweet text": "@HMRCcustomers No this is my first job", "ID": 0, "Label": 2, "text_label": "no complaint"}
print(dataset)
print("dataset[train][0]:", dataset["train"][0])
print("dataset[test][0]:", dataset["test"][0])


# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
#target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
#print("Target max length:", target_max_length)


# PREPROCESSING (uses tokenizer)
max_length = 64 # max_length of the input sequence

def preprocess_function(examples, text_column="Tweet text", label_column="text_label"):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
        labels["input_ids"][i] = [-100] * (max_length - len(label_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

processed_ds = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_ds["train"]
eval_dataset = processed_ds["test"]

batch_size = 16
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


# MODEL
model = AutoModelForCausalLM.from_pretrained(model_name)
print("model:", model)


# PEFT MODEL
peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=3)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # For example: "trainable params: 983,040 || all params: 560,197,632 || trainable%: 0.1754809274167014"


# TRAINING
# can also be done with the Transformers Trainer class
# https://huggingface.co/transformers/main_classes/trainer.html
lr = 3e-2
num_epochs = 3

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

for epoch in range(num_epochs):
    # training
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training epoch {epoch}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        #print("batch:", batch['input_ids'].shape)
        #print("batch:", batch['labels'].shape)

        #import ipdb
        #with ipdb.launch_ipdb_on_exception():
        outputs = model(**batch)

        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # evaluation
    """
    if epoch % 10 == 0:
        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Evaluation epoch {epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print("Evaluation epoch", epoch, ":")
        print("train_ppl:", train_ppl.item())
        print("train_epoch_loss:", train_epoch_loss.item())
        print("eval_ppl:", eval_ppl.item())
        print("eval_epoch_loss:", eval_epoch_loss.item())
    """

print("Optimized prompt:", model.get_prompt(batch_size=batch_size)[0][0].shape)

# INFERENCE
# Some examples
model.eval()
for i in range(2):
    text_column = "Tweet text"
    inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
    print("dataset[test][i][Tweet text]:", dataset["test"][i]["Tweet text"])

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        #print("inputs:", inputs)
        outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
        print("outputs", tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
