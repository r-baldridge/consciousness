#!/bin/bash
#
# Setup script for PII security hooks
# Run this after cloning the repository
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "🔒 Setting up PII security hooks..."

# 1. Configure git to use custom hooks
echo "Configuring git hooks path..."
git config core.hooksPath .githooks

# 2. Make hooks executable
echo "Making hooks executable..."
chmod +x .githooks/*

# 3. Create .pii-patterns if it doesn't exist
if [ ! -f ".pii-patterns" ]; then
    echo "Creating .pii-patterns from template..."
    cp .pii-patterns.example .pii-patterns
    echo ""
    echo "⚠️  IMPORTANT: Edit .pii-patterns to add your personal patterns!"
    echo "   This file is gitignored and won't be committed."
fi

# 4. Configure anonymous git identity (if not already set)
current_email=$(git config user.email || echo "")
if [[ ! "$current_email" =~ noreply ]]; then
    echo ""
    echo "⚠️  Your git email doesn't use noreply format."
    echo "   Current: $current_email"
    echo ""
    read -p "Would you like to set up an anonymous git identity? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your GitHub username: " username
        git config user.name "$username"
        git config user.email "$username@users.noreply.github.com"
        echo "✓ Git identity set to: $username <$username@users.noreply.github.com>"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Security hooks configured successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "The pre-commit hook will now scan for PII before each commit."
echo ""
echo "Next steps:"
echo "  1. Edit .pii-patterns to add your personal identifiers"
echo "  2. Test with: git add -A && git commit --dry-run"
echo ""
