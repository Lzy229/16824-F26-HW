import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

files = glob.glob("runs/experiment2/mAP.csv")

plt.figure(figsize=(10, 6))

line_styles = ['--', '-.', ':']

for i, file in enumerate(files):
    df = pd.read_csv(file)

    # Suppose your CSV has columns: x and y
    plt.plot(df["Step"], df["Value"], linestyle=line_styles[i % len(line_styles)],label=os.path.basename(file[:-4]))

plt.xlabel("Step")
plt.ylabel("mAP")
plt.title("mAP")
plt.legend(fontsize=8)
plt.grid(True)
save_path = "runs/experiment2/map.png"

plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()