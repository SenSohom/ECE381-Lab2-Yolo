"""
patch_builder.py — Patches llava/model/builder.py for Jetson Orin Nano (CUDA)

Run this ONCE after cloning the apple/ml-fastvlm repo:
    python3 patch_builder.py

What it fixes:
  1. LlavaQwen2ForCausalLM.from_pretrained()
       - Removes low_cpu_mem_usage=True  (triggers accelerate meta-device OOM)
       - Removes device_map from kwargs  (prevents accelerate from touching placement)
       - Forces torch_dtype=torch.float16
       - Adds model.cuda() after CPU load

  2. model.resize_token_embeddings()
       - Guards the call so it only runs if vocab size actually changed
       - Unconditional resize allocates on CUDA and crashes on stressed allocator
"""

import os
import sys

BUILDER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "llava", "model", "builder.py"
)

if not os.path.exists(BUILDER_PATH):
    print(f"ERROR: builder.py not found at {BUILDER_PATH}")
    print("Make sure you run this script from inside the ml-fastvlm directory.")
    sys.exit(1)

with open(BUILDER_PATH, "r") as f:
    content = f.read()

# ── Patch 1: LlavaQwen2ForCausalLM.from_pretrained ──────────────────────────

OLD_LOAD = """                model = LlavaQwen2ForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )"""

NEW_LOAD = """                # Patched for Jetson: plain CPU load then manual .cuda()
                # Removes low_cpu_mem_usage=True and device_map to bypass
                # accelerate's meta-device path which OOMs on unified memory.
                _load_kwargs = {k: v for k, v in kwargs.items()
                               if k not in ('device_map', 'torch_dtype')}
                model = LlavaQwen2ForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=False,
                    torch_dtype=torch.float16,
                    **_load_kwargs
                )
                model = model.cuda()"""

# ── Patch 2: resize_token_embeddings ────────────────────────────────────────

OLD_RESIZE = "    model.resize_token_embeddings(len(tokenizer))"

NEW_RESIZE = """    # Patched for Jetson: only resize if vocab size actually changed.
    # Unconditional resize_token_embeddings allocates on CUDA and can crash
    # the allocator in stressed container sessions (repeated failed runs).
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))"""

# ── Apply patches ────────────────────────────────────────────────────────────

patched = False

if OLD_LOAD in content:
    content = content.replace(OLD_LOAD, NEW_LOAD)
    print("✅ Patch 1 applied: LlavaQwen2ForCausalLM.from_pretrained → CPU-first load")
    patched = True
else:
    print("⚠️  Patch 1 NOT applied: from_pretrained pattern not found.")
    print("   The repo may have been updated. Check builder.py manually.")

if OLD_RESIZE in content:
    content = content.replace(OLD_RESIZE, NEW_RESIZE)
    print("✅ Patch 2 applied: resize_token_embeddings → conditional guard")
    patched = True
else:
    print("⚠️  Patch 2 NOT applied: resize_token_embeddings pattern not found.")
    print("   The repo may have been updated. Check builder.py manually.")

if patched:
    with open(BUILDER_PATH, "w") as f:
        f.write(content)
    print(f"\nSaved patched builder.py to: {BUILDER_PATH}")
    print("You can verify with: sed -n '125,185p' llava/model/builder.py")
else:
    print("\nNo patches were applied. builder.py is unchanged.")
    print("If you already applied the patches manually, this is expected.")
