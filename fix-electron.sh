#!/bin/bash
# Quick fix for Electron installation issues

echo "🔧 Fixing Electron installation..."

# Method 1: Clean reinstall with npm
echo "Method 1: Using npm for Electron..."
npm install electron@31.3.0 --ignore-scripts=false

# Method 2: If that fails, try with different permissions
if [ $? -ne 0 ]; then
    echo "Method 2: Trying with sudo (if needed)..."
    sudo npm install -g electron@31.3.0
    # Create symlink
    ln -sf $(which electron) ./node_modules/.bin/electron
fi

# Method 3: Manual download (last resort)
if [ $? -ne 0 ]; then
    echo "Method 3: Manual download approach..."
    npx @electron/download 31.3.0
fi

echo "✅ Try running: pnpm start"
