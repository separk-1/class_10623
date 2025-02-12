import json
import matplotlib.pyplot as plt

# ✅ 실험 데이터 경로
gqa_loss_path = "out/gqa_experiment_2kv/train_losses.json"
mingpt_loss_path = "out/mingpt_experiment/train_losses.json"

# ✅ JSON 파일에서 학습 손실 데이터 불러오기
with open(gqa_loss_path, "r") as f:
    gqa_losses = json.load(f)[:200]  # 200 iteration만 사용

with open(mingpt_loss_path, "r") as f:
    mingpt_losses = json.load(f)[:200]  # 200 iteration만 사용

# ✅ Iteration 리스트 생성 (0~199)
iterations = list(range(200))

# ✅ 스타일 설정 (Times New Roman 적용)
plt.style.use("ggplot")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

# ✅ 그래프 그리기
plt.figure(figsize=(10, 5), dpi=150)

plt.plot(iterations, gqa_losses, label="GQA (2 Key Heads)", color="blue", linestyle="-", linewidth=2)
plt.plot(iterations, mingpt_losses, label="minGPT (Multi-Head Attention)", color="orange", linestyle="-", linewidth=2)

# ✅ 그래프 스타일 설정
plt.xlabel("Iteration", fontsize=14, fontweight="bold")
plt.ylabel("Loss", fontsize=14, fontweight="bold")
plt.title("Training Loss: GQA (2 Key Heads) vs minGPT", fontsize=16, fontweight="bold")
plt.legend(fontsize=12, loc="upper right", frameon=True)
plt.grid(True, linestyle="--", alpha=0.6)

# ✅ 그래프 저장 및 출력
plt.savefig("gqa_vs_mingpt_loss_plot.png", dpi=300, bbox_inches="tight")
plt.show()
