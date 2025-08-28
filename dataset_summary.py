import os
import csv
from collections import defaultdict

DATA_DIR = 'data'
SPLITS = ['train', 'validation', 'test']
OUT_MANIFEST = 'manifest.csv'

all_classes = set()
per_split_counts = defaultdict(int)
per_class_counts = defaultdict(lambda: defaultdict(int))
rows = []

for split in SPLITS:
    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"Warning: split folder not found: {split_dir}")
        continue
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        count = len(files)
        per_split_counts[split] += count
        per_class_counts[split][cls] = count
        all_classes.add(cls)
        for fname in files:
            path = os.path.join(split, cls, fname).replace('\\', '/')
            rows.append((split, cls, path))

# Create a consistent class->label mapping sorted alphabetically
classes_sorted = sorted(list(all_classes))
class_to_label = {c: i for i, c in enumerate(classes_sorted)}

# Write manifest CSV with label indices
with open(OUT_MANIFEST, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['split', 'class', 'label', 'filepath'])
    for split, cls, path in rows:
        writer.writerow([split, cls, class_to_label[cls], path])

# Print summary
print('Dataset summary:')
print('  Total classes:', len(classes_sorted))
print('  Classes:', ', '.join(classes_sorted))
for split in SPLITS:
    print(f"  {split}: {per_split_counts.get(split,0)} images")
    # show top 5 classes by count in this split
    cls_counts = per_class_counts.get(split, {})
    if cls_counts:
        top5 = sorted(cls_counts.items(), key=lambda x: -x[1])[:5]
        print('    Top classes:', ', '.join([f"{c}({n})" for c,n in top5]))

print('\nManifest written to', OUT_MANIFEST)
