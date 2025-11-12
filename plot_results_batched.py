# plot_results_batched.py
import json
import matplotlib.pyplot as plt

with open("results_batched.json") as f:
    data = json.load(f)

methods = ["Full Batch", "Rare-First Batch"]
tokens_per_doc = [data["full_batch"]["tokens_per_doc"], data["rare_first_batch"]["input_tokens_per_doc"]]
time_per_doc = [data["full_batch"]["time_per_doc_sec"], data["rare_first_batch"]["time_per_doc_sec"]]

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.bar(methods, tokens_per_doc, color=['#95a5a6', '#27ae60'], alpha=0.8)
ax1.set_ylabel('Input Tokens per Doc', fontsize=12)

ax2 = ax1.twinx()
ax2.plot(methods, time_per_doc, 'r-o', linewidth=3, markersize=8)
ax2.set_ylabel('Time per Doc (s)', color='red', fontsize=12)

plt.title("AI Speed Reading Challenge\nBatched: 30×+ Faster", fontsize=14, fontweight='bold')
plt.suptitle(f"Speedup: {data['speedup']}× | Compute Saved: {data['compute_saved_percent']}%", 
             fontsize=12, color='green', y=0.88)
fig.legend(['Tokens', 'Time'], loc="upper center")
plt.tight_layout()
plt.savefig("speed_reading_batched.png", dpi=200)
print("Plot saved: speed_reading_batched.png")