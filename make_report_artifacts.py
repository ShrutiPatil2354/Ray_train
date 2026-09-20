import csv
import os
import matplotlib.pyplot as plt


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.abspath("ray_train_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


csv_path = os.path.join(OUTPUT_DIR, "training_metrics.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError("Run ray_train_demo.py before generating report artifacts.")

with open(csv_path, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise ValueError("The training metrics CSV is empty.")

epochs = [int(row["epoch"]) for row in rows]
losses = [float(row["loss"]) for row in rows]


print("=" * 60)
print("CSV LOADED")
print("=" * 60)
print(csv_path)


# ============================================================
# 2. CREATE LOSS GRAPH
# ============================================================

graph_path = os.path.join(
    OUTPUT_DIR,
    "training_loss_graph.png"
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    losses,
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Training Loss"
)

plt.title(
    "Ray Train - Training Loss vs Epoch"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    graph_path,
    dpi=300
)

plt.close()


print()
print("=" * 60)
print("GRAPH CREATED")
print("=" * 60)

print(graph_path)


# ============================================================
# 3. SUMMARY
# ============================================================

print()
print("=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

print(
    f"Number of epochs : {len(epochs)}"
)

print(
    f"Initial loss     : {losses[0]:.6f}"
)

print(
    f"Final loss       : {losses[-1]:.6f}"
)

print(
    f"Loss reduction   : {losses[0] - losses[-1]:.6f}"
)

print()
print("=" * 60)
print("ALL REPORT ARTIFACTS READY")
print("=" * 60)