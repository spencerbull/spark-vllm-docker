#!/bin/bash
set -euo pipefail

PATCH_URL="https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/39084.diff"
FALLBACK_URL="https://huggingface.co/bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16/raw/main/gemma4_patched.py"
TARGET="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py"
PATCH_FILE="/tmp/gemma4-nvfp4-pr39084.diff"

cd /usr/local/lib/python3.12/dist-packages

if grep -q "_map_gemma4_expert_param_name" "$TARGET"; then
    echo "Gemma4 NVFP4 fix already present"
else
    echo "Applying Gemma4 NVFP4 fix (PR #39084)"
    if curl -fsL "$PATCH_URL" -o "$PATCH_FILE" && git apply --exclude="tests/*" "$PATCH_FILE"; then
        echo "- PR #39084 applied successfully"
    else
        echo "- PR #39084 could not be applied cleanly, using patched gemma4.py fallback"
        curl -fsL "$FALLBACK_URL" -o "$TARGET"
    fi
fi

python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py")
text = path.read_text()
old = '                name = re.sub(r"\\.experts\\.(\\d+)\\.", r".moe.experts.\\1.", name)\n'
new = (
    '                if ".moe.experts." not in name:\n'
    '                    name = re.sub(r"\\.experts\\.(\\d+)\\.", r".moe.experts.\\1.", name)\n'
)
if old in text and new not in text:
    text = text.replace(old, new)
    path.write_text(text)
    print("- Patched Gemma4 expert key remap for existing .moe.experts checkpoints")
else:
    print("- Gemma4 expert key remap already compatible")
PY

rm -f "$PATCH_FILE"
