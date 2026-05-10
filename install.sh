#!/bin/bash
set -euo pipefail

python -m pip install -U pip setuptools wheel packaging ninja

pip install -r requirements.txt

pip install flash_attn==2.7.4.post1 --no-build-isolation