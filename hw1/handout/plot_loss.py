import pandas as pd
import matplotlib.pyplot as plt

# ✅ 1. RoPE 모델의 두 단계 loss 데이터 불러오기
rope_loss_600 = pd.read_csv("out/rope_experiment_600_16/training_loss.csv")
rope_loss_1200 = pd.read_csv("out/rope_experiment_1200_256/training_loss.csv")

# ✅ 2. minGPT 모델의 두 단계 loss 데이터 불러오기
mingpt_loss_600 = pd.read_csv("out/mingpt_experiment_600_16/training_loss.csv")
mingpt_loss_1200 = pd.read_csv("out/mingpt_experiment_1200_256/training_loss.csv")

# ✅ 3. Iteration 값을 맞춰서 합치기 (600번째부터 이어지도록 조정)
rope_loss_1200['iteration'] += 600
mingpt_loss_1200['iteration'] += 600

# ✅ 4. 데이터프레임 합치기
rope_loss = pd.concat([rope_loss_600, rope_loss_1200], ignore_index=True)
mingpt_loss = pd.concat([mingpt_loss_600, mingpt_loss_1200], ignore_index=True)

# ✅ 5. 스타일 적용
plt.style.use("ggplot")

# ✅ 6. 글꼴 설정 (Times New Roman)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

# ✅ 7. 그래프 그리기
plt.figure(figsize=(12, 6), dpi=150)

plt.xlim(0, 1200)

plt.plot(rope_loss['iteration'], rope_loss['loss'], label="RoPE GPT Loss",
         color='green', linestyle='-', linewidth=1, alpha=0.7)
plt.plot(mingpt_loss['iteration'], mingpt_loss['loss'], label="minGPT Loss",
         color='orange', linestyle='-', linewidth=1, alpha=0.7)

# ✅ 8. 그래프 스타일 설정
plt.xlabel("Iteration", fontsize=16, fontweight='bold')
plt.ylabel("Loss", fontsize=16, fontweight='bold')
plt.title("Training Loss Comparison: RoPE vs minGPT", fontsize=18, fontweight='bold')

plt.legend(fontsize=14, loc='upper right', frameon=True)
plt.grid(True, linestyle="--", alpha=0.6)

# ✅ 9. 그래프 저장
plt.savefig("training_loss_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
