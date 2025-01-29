import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

df = pd.read_csv("./txt_distribution.csv")

mpl.rc('font', family='Times New Roman')

# 플롯 그리기
plt.figure(figsize=(18, 6))  # 좌우로 약간 긴 크기
plt.hist(df['Article Length'], bins=30, weights=df['Count'], edgecolor='black', color='#6495ED', alpha=0.8)
plt.title("Histogram of Article Lengths", fontsize=16)
plt.xlabel("Article Length", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(axis='y', alpha=0.6, linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()