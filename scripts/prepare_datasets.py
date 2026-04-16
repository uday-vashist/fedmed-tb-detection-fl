# scripts/prepare_dataset.py
"""
Combines TBX11K + Shenzhen + TB Chest into data/combined/healthy/ and data/combined/tb/
No capping — copies full dataset.
Undersampling handled at training time in data_loader.py.
Run once before training. Re-run: delete data/combined first.
"""

import os
import shutil
import glob

TBX11K_DIR   = "data/raw/tbx11k/images"   # flat folder, h*.png and tb*.png
SHENZHEN_DIR = "data/raw/shenzhen"         # subfolders: normal/ and tb/
TBCHEST_DIR  = "data/tbchest"              # subfolders: Normal/ and tuberculosis/

OUTPUT_DIR   = "data/combined"

healthy_out = os.path.join(OUTPUT_DIR, "healthy")
tb_out      = os.path.join(OUTPUT_DIR, "tb")
os.makedirs(healthy_out, exist_ok=True)
os.makedirs(tb_out, exist_ok=True)

counter = {"healthy": 1, "tb": 1}

def copy_file(src, split):
    dst_dir  = healthy_out if split == "healthy" else tb_out
    prefix   = "h" if split == "healthy" else "tb"
    ext      = os.path.splitext(src)[1].lower()
    dst_name = f"{prefix}_{counter[split]:04d}{ext}"
    dst      = os.path.join(dst_dir, dst_name)
    shutil.copy2(src, dst)
    counter[split] += 1

# ── TBX11K ────────────────────────────────────────────────────────
print("Processing TBX11K...")
tbx11k_healthy = glob.glob(os.path.join(TBX11K_DIR, "h*.png"))
tbx11k_tb      = glob.glob(os.path.join(TBX11K_DIR, "tb*.png"))

for f in tbx11k_healthy:
    copy_file(f, "healthy")
for f in tbx11k_tb:
    copy_file(f, "tb")

print(f"  TBX11K   — healthy: {len(tbx11k_healthy)}, tb: {len(tbx11k_tb)}")

# ── Shenzhen ──────────────────────────────────────────────────────
print("Processing Shenzhen...")
shenzhen_healthy = glob.glob(os.path.join(SHENZHEN_DIR, "normal", "*.png"))
shenzhen_tb      = glob.glob(os.path.join(SHENZHEN_DIR, "tb", "*.png"))

for f in shenzhen_healthy:
    copy_file(f, "healthy")
for f in shenzhen_tb:
    copy_file(f, "tb")

print(f"  Shenzhen — healthy: {len(shenzhen_healthy)}, tb: {len(shenzhen_tb)}")

# ── TB Chest X-ray Database ───────────────────────────────────────
print("Processing TB Chest...")
tbchest_healthy = glob.glob(os.path.join(TBCHEST_DIR, "Normal", "*.png"))
tbchest_tb      = glob.glob(os.path.join(TBCHEST_DIR, "tuberculosis", "*.png"))

for f in tbchest_healthy:
    copy_file(f, "healthy")
for f in tbchest_tb:
    copy_file(f, "tb")

print(f"  TB Chest — healthy: {len(tbchest_healthy)}, tb: {len(tbchest_tb)}")

# ── Summary ───────────────────────────────────────────────────────
print("\n✓ Done.")
print(f"  Healthy : {counter['healthy']-1} images → {healthy_out}")
print(f"  TB      : {counter['tb']-1} images → {tb_out}")
print(f"  Ratio   : 1 TB : {(counter['healthy']-1)/(counter['tb']-1):.1f} healthy")
print(f"  Note    : Undersampling handled at training time in data_loader.py")
