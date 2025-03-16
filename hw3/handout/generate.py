import os
import pickle, json
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import datasets
import argparse
import dataloader
from tqdm.auto import tqdm

class ModelSampler:
    def __init__(self, out_dir, init_from="resume", device="cuda", max_new_tokens=5, temperature=0.6, top_k=1, ckpt_last = False):
        self.out_dir = out_dir
        self.init_from = init_from
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.ckpt_last = ckpt_last

        # Initialize sampling as part of __init__
        self._initialize_sampling()

    def _initialize_sampling(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        dtype = 'bfloat16'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.float16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        self.test_dataset = datasets.load_dataset("rotten_tomatoes", split='test')

        if self.init_from == 'resume':
            ckpt_path = os.path.join(self.out_dir, 'ckpt_last.pt' if self.ckpt_last else 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            self.model = GPT(gptconf)
            self.model.load_state_dict(checkpoint['model'])
        elif self.init_from.startswith('gpt2'):
            self.model = GPT.from_pretrained(self.init_from, dict(dropout=0.0))
        else:
            print("Warning: Invalid Resume paramater!")
            return 

        self.model.eval()
        self.model.to(self.device)

        enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: enc.encode(s, allowed_special={""})
        self.decode = lambda l: enc.decode(l)

    def get_generation(self, prompt):
        prompt_ids = self.encode(prompt)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)[None, ...]

        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
                return self.decode(y[0].tolist())

    def predict_labels(self, dataset):
        pred_samples = {}
        for i, row in tqdm(enumerate(dataset), total=len(dataset)):
            prompt = dataloader.get_sentiment_prompt(row["text"])
            response = self.get_generation(prompt)
            pred_samples[i] = {'response': response, 'true label': row['label']}
            ################################################## TODO: ##################################################
            truncated_response =  response[len(prompt):].strip().lower() #TODO: Truncate response to get predicted label (hint: the response is a string which includes input prompt plus the generated label)
            if "positive" in truncated_response:# TODO: Check if "positive" is in the truncated response
                predicted_label = 1
            elif "negative" in truncated_response: #TODO: Check if "negative" is in the truncated response
                predicted_label = 0
            else:
                predicted_label = -1
            pred_samples[i]['predicted label'] = predicted_label
        return pred_samples

    def compute_accuracy(self, pred_samples):
        counter = 0
        for iter, sample in pred_samples.items():
            if sample['predicted label'] == sample['true label']:
                counter += 1
        accuracy = counter / len(self.test_dataset)
        return accuracy, counter

    def get_accuracy(self):
        pred_samples = self.predict_labels(self.test_dataset)
        accuracy, counter = self.compute_accuracy(pred_samples)

        pos_counter = sum(1 for sample in pred_samples.values() if sample['predicted label'] == 1)
        neg_counter = sum(1 for sample in pred_samples.values() if sample['predicted label'] == 0)

        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(f"{self.out_dir}/results.json", "w") as f:
                json.dump({"pos_counter": pos_counter, "neg_counter": neg_counter, 
                            "accuracy": accuracy, "counter": counter}, f)
            with open(f"{self.out_dir}/predictions.json", "w") as f2:
                f2.write(json.dumps(pred_samples, indent=4))

        return accuracy, pos_counter, neg_counter, counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get model accuracy from command line parameters.')
    parser.add_argument('--out_dir', type=str, help='Output directory for model checkpoints')
    parser.add_argument('--init_from', type=str, default='resume', help='Initialization method', choices=['resume', 'gpt2', "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
    parser.add_argument('--max_new_tokens', type=int, default=5, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for generation')
    parser.add_argument('--top_k', type=int, default=1, help='Top K tokens to sample from')
    args = parser.parse_args()

    sampler = ModelSampler(args.out_dir, args.init_from, args.device, args.max_new_tokens, args.temperature, args.top_k)
    accuracy, pos_counter, neg_counter, counter = sampler.get_accuracy()
    print(f"Best Val Checkpoint || Accuracy: {accuracy}, Positive Predictions: {pos_counter}, Negative Predictions: {neg_counter}, Correct Predictions: {counter}")
    sampler = ModelSampler(args.out_dir, args.init_from, args.device, args.max_new_tokens, args.temperature, args.top_k, ckpt_last=True)
    accuracy, pos_counter, neg_counter, counter = sampler.get_accuracy()
    print(f"Last Iter Checkpoint || Accuracy: {accuracy}, Positive Predictions: {pos_counter}, Negative Predictions: {neg_counter}, Correct Predictions: {counter}")