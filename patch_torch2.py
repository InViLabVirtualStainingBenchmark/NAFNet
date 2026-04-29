# patch_torch2.py
import os

files_to_patch = []
for root, dirs, files in os.walk("basicsr"):
    for f in files:
        if f.endswith(".py"):
            files_to_patch.append(os.path.join(root, f))

for filepath in files_to_patch:
    with open(filepath, "r") as f:
        content = f.read()

    original = content

    # Fix 1: amp.autocast
    content = content.replace(
        "torch.cuda.amp.autocast",
        "torch.amp.autocast('cuda')"
    )

    # Fix 2: torch._six removed in PyTorch 2.x
    content = content.replace(
        "from torch._six import string_classes",
        "string_classes = str"
    )
    content = content.replace(
        "torch._six.string_classes",
        "str"
    )

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Patched: {filepath}")

print("Patching complete.")