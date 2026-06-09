"""
View NAFNet inference results — side-by-side comparison of HE input, predicted IHC, and ground truth.

Usage:
    python scripts/view_results.py --dataset BCI
    python scripts/view_results.py --dataset BCI --num_images 5
    python scripts/view_results.py --dataset BCI --save --output_dir results/figures
"""

import os
import argparse
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description='View NAFNet inference results')
parser.add_argument('--dataset',    default='BCI', choices=['BCI', 'MIST_ER', 'MIST_HER2', 'MIST_Ki67', 'MIST_PR'])
parser.add_argument('--results_dir', default=None, help='Override results directory')
parser.add_argument('--num_images', default=3, type=int, help='Number of images to show')
parser.add_argument('--save',       action='store_true', help='Save figures instead of displaying')
parser.add_argument('--output_dir', default='results/figures', help='Where to save figures')
args = parser.parse_args()

# Default results path follows BasicSR convention
RESULTS_ROOTS = {
    'BCI':       'results/NAFNet-BCI-local-smoke/visualization/BCI_test',
    'MIST_ER':   'results/NAFNet-MIST-ER-local-smoke/visualization/MIST_ER_val',
    'MIST_HER2': 'results/NAFNet-MIST-HER2-local-smoke/visualization/MIST_HER2_val',
    'MIST_Ki67': 'results/NAFNet-MIST-Ki67-local-smoke/visualization/MIST_Ki67_val',
    'MIST_PR':   'results/NAFNet-MIST-PR-local-smoke/visualization/MIST_PR_val',
}

vis_dir = args.results_dir or RESULTS_ROOTS[args.dataset]

if not os.path.exists(vis_dir):
    print(f"Results directory not found: {vis_dir}")
    print("Run inference first with basicsr/test.py")
    exit(1)

# Collect predicted images (BasicSR saves {name}.png for prediction)
pred_files = sorted([
    f for f in os.listdir(vis_dir)
    if f.lower().endswith('.png') and not f.endswith('_gt.png') and not f.endswith('_he.png')
])[:args.num_images]

if not pred_files:
    print(f"No predicted images found in {vis_dir}")
    exit(1)

if args.save:
    os.makedirs(args.output_dir, exist_ok=True)

print(f"Showing {len(pred_files)} results from: {vis_dir}\n")

for fname in pred_files:
    base = os.path.splitext(fname)[0]

    pred_path = os.path.join(vis_dir, fname)
    gt_path   = os.path.join(vis_dir, f"{base}_gt.png")
    he_path   = os.path.join(vis_dir, f"{base}_he.png")

    pred_img = Image.open(pred_path).convert('RGB')
    w, h = pred_img.size

    panels = []
    labels = []

    if os.path.exists(he_path):
        panels.append(Image.open(he_path).convert('RGB'))
        labels.append('HE Input')

    panels.append(pred_img)
    labels.append('NAFNet Output')

    if os.path.exists(gt_path):
        panels.append(Image.open(gt_path).convert('RGB'))
        labels.append('IHC Ground Truth')

    n = len(panels)
    combined = Image.new('RGB', (w * n + 10 * (n - 1), h + 30), (255, 255, 255))

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(combined)

    for i, (panel, label) in enumerate(zip(panels, labels)):
        x_offset = i * (w + 10)
        combined.paste(panel, (x_offset, 30))
        draw.text((x_offset + w // 2 - len(label) * 6, 5), label, fill='black', font=font)

    if args.save:
        save_path = os.path.join(args.output_dir, f"comparison_{base}.png")
        combined.save(save_path)
        print(f"Saved: {save_path}")
    else:
        combined.show()
        print(f"Showing: {fname}")