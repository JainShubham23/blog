#!/usr/bin/env python3
"""
Essential Pelican blog management tasks
"""

import sys
import subprocess
import shutil
from pathlib import Path

def build():
    """Build the static site"""
    print("Building site...")
    result = subprocess.run(['pelican', 'content'], cwd=Path.cwd())
    if result.returncode == 0:
        print("✅ Site built successfully!")
    else:
        print("❌ Build failed!")
        sys.exit(1)

def publish():
    """Build the site for production"""
    print("Building site for production...")
    result = subprocess.run(['pelican', 'content', '-s', 'publishconf.py'], cwd=Path.cwd())
    if result.returncode == 0:
        print("✅ Production site built successfully!")
    else:
        print("❌ Production build failed!")
        sys.exit(1)

def deploy():
    """Deploy to GitHub Pages"""
    print("🚀 Deploying to GitHub Pages...")
    try:
        # Build for production
        publish()
        
        # Deploy using ghp-import
        subprocess.run(['ghp-import', '-m', 'Generate Pelican site', '-b', 'gh-pages', 'output'], check=True)
        
        # Push to GitHub
        subprocess.run(['git', 'push', 'origin', 'gh-pages'], check=True)
        
        print("🎉 Deployment successful!")
        print("📱 Site: https://jainshubham23.github.io/personal_blog")
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

def serve(port=8000):
    """Serve the site locally with auto-reload"""
    print(f"Starting server on http://localhost:{port}")
    try:
        subprocess.run(['pelican', '--listen', '--autoreload', '--port', str(port), 'content'])
    except KeyboardInterrupt:
        print("\nServer stopped.")

def clean():
    """Clean the output directory"""
    output_dir = Path('output')
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print("✅ Output directory cleaned!")
    else:
        print("Output directory doesn't exist.")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tasks.py build     - Build the site")
        print("  python tasks.py publish   - Build for production")
        print("  python tasks.py deploy    - Deploy to GitHub Pages")
        print("  python tasks.py serve     - Serve locally with auto-reload")
        print("  python tasks.py clean     - Clean output directory")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'build':
        build()
    elif command == 'publish':
        publish()
    elif command == 'deploy':
        deploy()
    elif command == 'serve':
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        serve(port)
    elif command == 'clean':
        clean()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == '__main__':
    main()