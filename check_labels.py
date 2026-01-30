import glob, os

txt_dir = "/home/anirudhaseetiraju/coding-workstation/severstal_yolo/labels/train"
txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))

print(f"Found {len(txt_files)} .txt files")

bad_count = 0
for txt in txt_files[:20]:  # check first 20
    with open(txt, 'r') as f:
        lines = f.readlines()
    if not lines:
        print(f"Empty file: {os.path.basename(txt)}")
        bad_count += 1
        continue
    for i, line in enumerate(lines[:2], 1):
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls = int(parts[0])  # must succeed for integer class
            coords = [float(p) for p in parts[1:]]
            if len(coords) % 2 != 0:
                print(f"Odd number of coords in {os.path.basename(txt)} line {i}")
                bad_count += 1
            if any(c < 0 or c > 1 for c in coords):
                print(f"Out-of-range coord in {os.path.basename(txt)} line {i}")
                bad_count += 1
        except ValueError:
            print(f"Invalid class or float parse in {os.path.basename(txt)} line {i}: {line.strip()}")
            bad_count += 1

print(f"Checked 20 files; found {bad_count} issues in samples.")