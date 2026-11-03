#!/bin/bash
# Push any changes you make later:  bash update.sh "what you changed"
set -e
cd "$(dirname "$0")"
MSG=${1:-"Update project"}
git add .
if git ls-files | grep -q "secrets.toml$"; then
  echo "Safety stop: key file detected. Nothing was pushed."; exit 1
fi
git commit -m "$MSG"
git push
echo "Done. Changes are on GitHub."
