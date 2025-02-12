import json
import pandas as pd

# JSON 파일 불러오기

filename = "gqa_experiment_1200_256"
with open(f"out/{filename}/train_losses.json", "r") as f:
    loss_data = json.load(f)

# DataFrame 변환
df = pd.DataFrame({"iteration": range(1, len(loss_data) + 1), "loss": loss_data})

# CSV로 저장
df.to_csv(f"out/{filename}/training_loss.csv", index=False)

print(f"CSV saved: out/{filename}/training_loss.csv")
