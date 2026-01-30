import json
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
csv_path = '/home/anirudhaseetiraju/coding-workstation/train.csv'  # Update to your original train.csv
images_dir = '/home/anirudhaseetiraju/coding-workstation/train_images'
output_json = '/home/anirudhaseetiraju/coding-workstation/instances_train.json'

# Load original CSV
df = pd.read_csv(csv_path)

# Get the images (from filenames)
selected_stems = {Path(f).stem for f in os.listdir(images_dir) if f.endswith('.jpg')}

# Filter CSV to only selected images
df_small = df[df['ImageId'].str.replace('.jpg', '', regex=False).isin(selected_stems)]

# COCO structure
coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "defect_1"},
        {"id": 2, "name": "defect_2"},
        {"id": 3, "name": "defect_3"},
        {"id": 4, "name": "defect_4"}
    ]
}

ann_id = 1
for img_id, row in df_small.iterrows():
    img_name = row['ImageId']
    img_path = os.path.join(images_dir, img_name)
    if not os.path.exists(img_path):
        continue
    
    # Add image entry
    coco["images"].append({
        "id": img_id,
        "file_name": img_name,
        "width": 1600,
        "height": 256
    })
    
    # Decode RLE to bbox (simple min-max for bounding box)
    if pd.notna(row['EncodedPixels']):
        mask_rle = row['EncodedPixels']
        s = mask_rle.split()
        starts, lengths = [int(x) for x in s[::2]], [int(x) for x in s[1::2]]
        min_x, min_y, max_x, max_y = 1600, 256, 0, 0
        for start, length in zip(starts, lengths):
            y = (start // 1600)
            x = start % 1600
            for i in range(length):
                curr_x = (start + i) % 1600
                curr_y = (start + i) // 1600
                min_x = min(min_x, curr_x)
                max_x = max(max_x, curr_x)
                min_y = min(min_y, curr_y)
                max_y = max(max_y, curr_y)
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        if width > 0 and height > 0:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": row['ClassId'],
                "bbox": [min_x, min_y, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            ann_id += 1

# Save JSON
with open(output_json, 'w') as f:
    json.dump(coco, f, indent=2)

print(f"COCO annotations created: {output_json}")
print(f"Images: {len(coco['images'])} | Annotations: {len(coco['annotations'])}")