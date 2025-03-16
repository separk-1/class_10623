from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

def get_sentiment_prompt(text, label=''):
    #TODO: For Question 5.11, try a different prompt template for better/worse performance.
    instruction = "Please read the following text and classify it as either positive or negative sentiment. Remember to consider the overall tone, context, and emotional cues conveyed in the text. Positive sentiments generally express happiness, satisfaction, or positivity, while negative sentiments convey sadness, anger, or negativity."

    INSTRUCTION_TEMPLATE = "Instruction: {}\nText: {}\nLabel: {}"
    return INSTRUCTION_TEMPLATE.format(instruction, text, label) 

class CustomDataLoader:
    def __init__(self, dataset,  tokenizer, batch_size=8):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt", padding="max_length", max_length=1024)
        
        # Format dataset with prompts and answers
        self.formatted_dataset = dataset.map(self._add_instruction_finetuning, remove_columns=dataset.column_names, load_from_cache_file=False)
        self.formatted_dataset.set_format(type='torch', columns=['instr_tuned_text', 'instr_len'])

    def _add_instruction_finetuning(self, rec):
        # Convert label from 0/1 to "negative"/"positive"
        label = "positive" if rec["label"] == 1 else "negative"
        text_with_prompt = get_sentiment_prompt(rec["text"], label)

        # Find "Label:" position and tokenize up to that point
        label_pos = text_with_prompt.find("Label:")
        token_len = 0
        if label_pos != -1:
            instructions = text_with_prompt[:label_pos]
            instructions_tokens = self.tokenizer(
                instructions, add_special_tokens=False, truncation=True, max_length=256
            ).input_ids
            token_len = len(instructions_tokens)

        # Store text + instructions length
        rec["instr_tuned_text"] = text_with_prompt
        rec["instr_len"] = token_len
        return rec

    def _tokenize(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding=True, max_length = 1024)  # Dynamic padding will be applied later

    def collate_fn(self, batch):
        texts = [item["instr_tuned_text"] for item in batch]
        tokenized_batch = self.tokenizer(texts, truncation=True, padding=True, 
                                        max_length=1024, return_tensors="pt")
        input_ids = tokenized_batch["input_ids"]
        labels = input_ids[:, 1:].clone()
        labels = torch.cat(
            [labels, torch.full((labels.size(0), 1), self.tokenizer.pad_token_id, dtype=torch.long)],
            dim=1
        )
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Use the precomputed instruction lengths to replace labels
        for i in range(labels_padded.size(0)):
            instr_len = batch[i]["instr_len"]
            if instr_len > 0:
                labels_padded[i, :instr_len] = -100 

        return input_ids_padded, labels_padded

    def get_loader(self, shuffle=True):
        return DataLoader(self.formatted_dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
