import json
import numpy as np
import matplotlib.pyplot as plt

# ✅ key head 개수별 experiment 폴더 설정
experiments = {
    1: "out/gqa_experiment_1kv",
    2: "out/gqa_experiment_2kv",
    3: "out/gqa_experiment_3kv",
    6: "out/gqa_experiment_6kv"
}

# ✅ 평균 실행 시간 저장할 리스트
key_heads = []
avg_times = []

# ✅ 각 실험 폴더에서 실행 시간 데이터 읽기
for key_head, folder in experiments.items():
    file_path = f"{folder}/attention_computation_time.json"
    
    with open(file_path, "r") as f:
        attn_times = json.load(f)  # iteration별 실행 시간 리스트
    
    # ✅ 평균 실행 시간 계산
    avg_time = np.mean(attn_times)
    
    # ✅ 리스트에 저장
    key_heads.append(key_head)
    avg_times.append(avg_time)

# ✅ 스타일 설정 (Times New Roman 적용)
plt.style.use("ggplot")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

# ✅ 그래프 그리기
plt.figure(figsize=(8, 5), dpi=150)
plt.plot(key_heads, avg_times, marker="o", linestyle="-", color="b", linewidth=2, markersize=8)

# ✅ 그래프 스타일 설정
plt.xlabel("Number of Key Heads", fontsize=14, fontweight="bold")
plt.ylabel("Average Attention Time (ms)", fontsize=14, fontweight="bold")
plt.title("Average Attention Computation Time", fontsize=16, fontweight="bold")
plt.xticks(key_heads)
plt.grid(True, linestyle="--", alpha=0.6)

# ✅ 그래프 저장 및 출력
plt.savefig("average_attention_time_plot.png", dpi=300, bbox_inches="tight")
plt.show()
