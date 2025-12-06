#!/usr/bin/env python3
"""
Quick script to run the Streamlit app locally.

Usage:
    python run_app.py
    
Or directly:
    streamlit run app/Home.py
"""

import subprocess
import sys

def main():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "app/Home.py",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main()