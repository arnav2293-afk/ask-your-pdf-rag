#!/bin/bash
# One-command setup for Ask Your PDF (Mac).
# Run inside VS Code's terminal:  bash setup.sh
set -e
cd "$(dirname "$0")"
REPO_URL="https://github.com/arnav2293-afk/ask-your-pdf-rag.git"

echo ""
echo "=== Step 1 of 6: checking Python ==="
PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "Found Python $PYV"
case "$PYV" in
  3.10|3.11|3.12) echo "Good version." ;;
  *) echo "Please install Python 3.12 from python.org, restart Terminal, and run this again."; exit 1 ;;
esac

echo ""
echo "=== Step 2 of 6: creating virtual environment ==="
[ -d venv ] || python3 -m venv venv
source venv/bin/activate

echo ""
echo "=== Step 3 of 6: installing libraries (takes 5 to 10 minutes) ==="
pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "=== Step 4 of 6: saving your Gemini API key ==="
if [ -f .streamlit/secrets.toml ]; then
  echo "Key file already exists, skipping."
else
  read -s -p "Paste your Gemini API key (it stays hidden) and press Enter: " KEY; echo
  printf 'GEMINI_API_KEY = "%s"\n' "$KEY" > .streamlit/secrets.toml
  echo "Saved to .streamlit/secrets.toml (this file never goes to GitHub)."
fi

echo ""
echo "=== Step 5 of 6: uploading to GitHub ==="
if [ -z "$(git config --global user.email)" ]; then
  read -p "Type the email you use on GitHub and press Enter: " GHEMAIL
  git config --global user.name "Arnav Shahi"
  git config --global user.email "$GHEMAIL"
fi
[ -d .git ] || git init -q
git add .
if git ls-files | grep -q "secrets.toml$"; then
  echo "Safety stop: your key file was about to be uploaded. Nothing was pushed."; exit 1
fi
git commit -q -m "Ask Your PDF: hybrid RAG chatbot with page citations" || echo "Nothing new to commit."
git branch -M main
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"
git push -u origin main
echo "Uploaded! Open https://github.com/arnav2293-afk/ask-your-pdf-rag to check."

echo ""
echo "=== Step 6 of 6: starting the app ==="
echo "Your browser will open. Press Control + C here to stop the app."
streamlit run app.py
