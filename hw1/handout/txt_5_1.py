import torch
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN

# ✅ 학습된 모델 불러오기
model_path = "out/rope_experiment_600_16/model.pt"
config_path = "out/rope_experiment_600_16/config.json"

config = CN(load_json=config_path)
model = GPT(config.model)
model.load_state_dict(torch.load(model_path))
model.eval()

# ✅ 입력 문장 (Shakespeare 첫 문장)
context = "O God, O God!"
encoded_context = [train_dataset.stoi[s] for s in context]

x = torch.tensor(encoded_context, dtype=torch.long)[None, ...]
y, attn_time = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)

# ✅ 생성된 텍스트 출력
print("Generated text:\n", ''.join([train_dataset.itos[int(i)] for i in y[0]]))

