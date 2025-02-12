import pandas as pd
import matplotlib.pyplot as plt

# ✅ 1. 네 가지 설정의 두 단계 loss 데이터 불러오기
mingpt_loss_600 = pd.read_csv("out/mingpt_experiment_600_16/training_loss.csv")
mingpt_loss_1200 = pd.read_csv("out/mingpt_experiment_1200_256/training_loss.csv")

rope_loss_600 = pd.read_csv("out/rope_experiment_600_16/training_loss.csv")
rope_loss_1200 = pd.read_csv("out/rope_experiment_1200_256/training_loss.csv")

gqa_loss_600 = pd.read_csv("out/gqa_experiment_600_16/training_loss.csv")
gqa_loss_1200 = pd.read_csv("out/gqa_experiment_1200_256/training_loss.csv")

rope_gqa_loss_600 = pd.read_csv("out/rope_gqa_experiment_600_16/training_loss.csv")
rope_gqa_loss_1200 = pd.read_csv("out/rope_gqa_experiment_1200_256/training_loss.csv")

# ✅ 2. Iteration 값을 맞춰서 합치기 (600번째부터 이어지도록 조정)
mingpt_loss_1200['iteration'] += 600
rope_loss_1200['iteration'] += 600
gqa_loss_1200['iteration'] += 600
rope_gqa_loss_1200['iteration'] += 600

# ✅ 3. 데이터프레임 합치기
mingpt_loss = pd.concat([mingpt_loss_600, mingpt_loss_1200], ignore_index=True)
rope_loss = pd.concat([rope_loss_600, rope_loss_1200], ignore_index=True)
gqa_loss = pd.concat([gqa_loss_600, gqa_loss_1200], ignore_index=True)
rope_gqa_loss = pd.concat([rope_gqa_loss_600, rope_gqa_loss_1200], ignore_index=True)

# ✅ 4. 그래프 스타일 설정
plt.style.use("ggplot")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

# ✅ 5. 그래프 그리기 (색상 변경 + 실선 적용)
plt.figure(figsize=(12, 6), dpi=150)

plt.plot(mingpt_loss['iteration'], mingpt_loss['loss'], label="minGPT (No RoPE, No GQA)",
         color='#E74C3C', linestyle='-', linewidth=1, alpha=0.8)  # 빨간색

plt.plot(rope_loss['iteration'], rope_loss['loss'], label="RoPE only",
         color='#3498DB', linestyle='-', linewidth=1, alpha=0.8)  # 파란색

plt.plot(gqa_loss['iteration'], gqa_loss['loss'], label="GQA only",
         color='#27AE60', linestyle='-', linewidth=1, alpha=0.8)  # 초록색

plt.plot(rope_gqa_loss['iteration'], rope_gqa_loss['loss'], label="RoPE + GQA",
         color='#8E44AD', linestyle='-', linewidth=1, alpha=0.8)  # 보라색

# ✅ 6. 그래프 제목 & 축 설정
plt.xlabel("Iteration", fontsize=16, fontweight="bold")
plt.ylabel("Loss", fontsize=16, fontweight="bold")
plt.title("Training Loss Comparison", fontsize=18, fontweight="bold")

plt.legend(fontsize=14, loc="upper right", frameon=True)
plt.grid(True, linestyle="--", alpha=0.6)

# ✅ 7. 그래프 저장
plt.savefig("training_loss_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
