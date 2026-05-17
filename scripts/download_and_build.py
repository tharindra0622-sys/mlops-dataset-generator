#!/usr/bin/env python3
"""
Download all run artifacts from GitHub Actions and build the master dataset.
Run this on your local machine after enough runs have accumulated.

Requirements:
    pip install requests pandas

Usage:
    export GITHUB_TOKEN=your_personal_access_token
    python scripts/download_and_build.py --repo YOUR_USERNAME/mlops-dataset-generator
"""

import os
import sys
import json
import zipfile
import argparse
import requests
import pandas as pd
from pathlib import Path
from io import BytesIO

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
BASE_URL     = "https://api.github.com"

def github_get(url, token):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r

def list_artifacts(repo, token, per_page=100):
    url = f"{BASE_URL}/repos/{repo}/actions/artifacts?per_page={per_page}"
    all_artifacts = []
    while url:
        r = github_get(url, token)
        data = r.json()
        all_artifacts.extend(data.get("artifacts", []))
        url = r.links.get("next", {}).get("url")
    return all_artifacts

def download_artifact(artifact, repo, token, out_dir):
    artifact_id   = artifact["id"]
    artifact_name = artifact["name"]
    download_url  = f"{BASE_URL}/repos/{repo}/actions/artifacts/{artifact_id}/zip"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    r = requests.get(download_url, headers=headers, stream=True)
    if r.status_code != 200:
        print(f"  [WARN] Could not download {artifact_name}: {r.status_code}")
        return

    out_path = Path(out_dir) / artifact_name
    out_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(BytesIO(r.content)) as z:
        z.extractall(out_path)

    print(f"  Downloaded: {artifact_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="GitHub repo in format owner/repo")
    parser.add_argument("--out", default="data/all_logs",
                        help="Output directory for artifacts")
    parser.add_argument("--max", type=int, default=500,
                        help="Maximum number of artifacts to download")
    args = parser.parse_args()

    if not GITHUB_TOKEN:
        print("ERROR: Set GITHUB_TOKEN environment variable")
        sys.exit(1)

    print(f"Fetching artifacts from {args.repo}...")
    artifacts = list_artifacts(args.repo, GITHUB_TOKEN)

    # Filter to only run-log artifacts
    run_logs = [a for a in artifacts if a["name"].startswith("run-log-")]
    print(f"Found {len(run_logs)} run-log artifacts")
    run_logs = run_logs[:args.max]

    Path(args.out).mkdir(parents=True, exist_ok=True)

    for i, artifact in enumerate(run_logs):
        print(f"[{i+1}/{len(run_logs)}] ", end="")
        download_artifact(artifact, args.repo, GITHUB_TOKEN, args.out)

    print(f"\nAll artifacts downloaded to {args.out}")
    print("Now run: python scripts/collect_dataset.py")

if __name__ == "__main__":
    main()
