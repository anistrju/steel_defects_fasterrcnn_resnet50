import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import shutil


def rle_decode(mask_rle, shape=(1600, 256)):
    """Decode RLE to binary mask."""
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Transpose to match image orientation

def mask_to_polygons(mask):
    """Convert binary mask to normalized polygons for YOLO segmentation."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for contour in contours:
        if len(contour) < 3:
            continue
            
        # Convert to float and reshape to (N, 2)
        points = contour.astype(np.float32).reshape(-1, 2)
        
        # Normalize
        points[:, 0] /= mask.shape[1]   # width  (columns)
        points[:, 1] /= mask.shape[0]   # height (rows)
        
        # Flatten back to 1D list for YOLO format
        flat = points.flatten().tolist()
        
        # YOLO requires at least 6 values (3 points)
        if len(flat) >= 6:
            polygons.append(flat)
    
    return polygons


def convert_to_yolo(csv_path, images_dir, output_dir, split_ratio=0.8):
    #clean up from earlier runs
    output_root = os.path.join(output_dir, 'images')
    labels_root = os.path.join(output_dir, 'labels')
    
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    if os.path.exists(labels_root):
        shutil.rmtree(labels_root)
    
    # Now recreate empty structure
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'),   exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'),   exist_ok=True)

    df = pd.read_csv(csv_path)
    unique_images = df['ImageId'].unique()
    
    # Split train/val
    train_images, val_images = train_test_split(unique_images, train_size=split_ratio, random_state=42)
    
    for split, images in [('train', train_images), ('val', val_images)]:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
        for img_name in images:
            # Copy image
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(output_dir, 'images', split, img_name)
            if os.path.exists(src_img):
                os.symlink(src_img, dst_img)  # Or shutil.copy for full copy
            
            # Get all defects for this image
            img_df = df[df['ImageId'] == img_name]
            labels = []
            for _, row in img_df.iterrows():
                if not pd.isna(row['EncodedPixels']):
                    mask = rle_decode(row['EncodedPixels'])
                    polygons = mask_to_polygons(mask)
                    class_id = int(row['ClassId'] - 1)  # 0-based
                    for poly in polygons:
                        if len(poly) >= 6:  # At least 3 points
                            # Format class as INTEGER, coordinates as %.6f
                            label_line = [str(class_id)] + [f"{x:.6f}" for x in poly]
                            labels.append(' '.join(label_line))
                        else:
                            print(f"Skipped small polygon in {img_name}: only {len(poly)//2} points")
                            
            
            # Write label TXT
            if labels:
                label_path = os.path.join(output_dir, 'labels', split, img_name.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write('\n'.join(labels)+ '\n')
                print(f"Wrote {len(labels)} instances to {label_path}")

# Usage
convert_to_yolo(
    csv_path='train.csv',
    images_dir='train_images',
    output_dir='severstal_yolo',
    split_ratio=0.8
)
print("Conversion complete. Update severstal.yaml with the output_dir.")