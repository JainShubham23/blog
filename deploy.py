#!/usr/bin/env python3
"""
Deploy script for GitHub Pages
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def deploy():
    """Deploy to GitHub Pages"""
    print("🚀 Starting deployment to GitHub Pages...")
    
    # Build the site for production
    if not run_command("pelican content -s publishconf.py", "Building site for production"):
        return False
    
    # Deploy to GitHub Pages
    if not run_command("ghp-import -m 'Generate Pelican site' -b gh-pages output", "Importing to gh-pages branch"):
        return False
    
    # Push to GitHub
    if not run_command("git push origin gh-pages", "Pushing to GitHub Pages"):
        return False
    
    print("🎉 Deployment completed successfully!")
    print("📱 Your site will be available at: https://jainshubham23.github.io/personal_blog")
    print("⏳ It may take a few minutes for changes to appear.")

if __name__ == "__main__":
    deploy()