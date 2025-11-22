import os
import re
import shutil

# ====== USER SETTINGS ======
INPUT_DIR  = r"C:\Users\YJL3090\PycharmProjects\Multiview-3D-Reconstruction\yunnangarden_sculpture"
N_SELECT   = 10   # how many images you want
# ===========================

# Output folder automatically named using N_SELECT
OUTPUT_DIR = os.path.join(INPUT_DIR, f"subset_{N_SELECT}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def numeric_key(name: str):
    """
    Extracts last integer in filename for sorting: sfm0.jpg, sfm1.jpg, ..., sfm142.jpg
    """
    nums = re.findall(r'\d+', name)
    return int(nums[-1]) if nums else -1

# Get all image files
all_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

if not all_files:
    raise RuntimeError("No image files found in INPUT_DIR.")

# Sort by numeric index so sfm2 < sfm10 < sfm100
all_files = sorted(all_files, key=numeric_key)

n_total = len(all_files)
print(f"Found {n_total} images.")

# If fewer than requested, just copy all
if n_total <= N_SELECT:
    selected_indices = list(range(n_total))
else:
    # Evenly spaced indices from 0 to n_total-1
    step = (n_total - 1) / (N_SELECT - 1)
    selected_indices = sorted({int(round(i * step)) for i in range(N_SELECT)})

print("Selected indices:", selected_indices)

selected_names = []

# Copy selected images
for idx in selected_indices:
    src_name = all_files[idx]
    src = os.path.join(INPUT_DIR, src_name)
    dst = os.path.join(OUTPUT_DIR, src_name)
    print(f"Copying {src} -> {dst}")
    shutil.copy2(src, dst)
    selected_names.append(src_name)

# ---- Copy K.txt if it exists ----
k_src = os.path.join(INPUT_DIR, "K.txt")
if os.path.exists(k_src):
    k_dst = os.path.join(OUTPUT_DIR, "K.txt")
    shutil.copy2(k_src, k_dst)
    print(f"Copied K.txt -> {k_dst}")
else:
    print("[WARN] K.txt not found in INPUT_DIR; skipping.")

# Save list of selected files in a text file inside the output folder
list_path = os.path.join(OUTPUT_DIR, f"selected_{N_SELECT}.txt")
with open(list_path, "w", encoding="utf-8") as f:
    for name in selected_names:
        f.write(name + "\n")

print(f"Done. Saved {len(selected_indices)} images to: {OUTPUT_DIR}")
print(f"List of selected images written to: {list_path}")
