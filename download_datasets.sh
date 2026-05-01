#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/datasets/wikivoyage-listings-en.csv"

curl -L "https://raw.githubusercontent.com/wikivoyage/wikivoyage.github.io/refs/heads/master/wikivoyage-listings-en.csv" -o "$DEST"
echo "Saved to $DEST"
