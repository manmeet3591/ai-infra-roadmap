#!/bin/bash
# =====================================================================
# setup_and_push.sh
# Run this script on your local machine to publish ai-infra-roadmap.
# Prereqs: git, gh CLI (authenticated), Python 3.10+
# =====================================================================

set -e

REPO="ai-infra-roadmap"
GITHUB_USER="manmeet3591"
PAGES_URL="https://${GITHUB_USER}.github.io/${REPO}"

echo "=== AI Infra Roadmap: GitHub Setup ==="

# 1. Init git
git init
git add .
git commit -m "Initial commit: AI Infrastructure Engineering roadmap (12 stages)"

# 2. Create public GitHub repo
gh repo create "${GITHUB_USER}/${REPO}" \
  --public \
  --description "Hands-on AI Infrastructure Engineering journal — CUDA kernels to multi-node serving" \
  --push \
  --source .

# 3. Enable GitHub Pages
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "/repos/${GITHUB_USER}/${REPO}/pages" \
  -f source='{"branch":"gh-pages","path":"/"}' \
  2>/dev/null || echo "(Pages will auto-enable on first Actions run)"

echo ""
echo "=== Done! ==="
echo "Repo:  https://github.com/${GITHUB_USER}/${REPO}"
echo "Pages: ${PAGES_URL}  (live in ~2 minutes after Actions run)"
echo ""
echo "To preview locally first:"
echo "  pip install -r requirements.txt"
echo "  mkdocs serve"
