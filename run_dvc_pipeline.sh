#!/usr/bin/env bash
set -euo pipefail

git init
git config user.email "student@example.com"
git config user.name "student"

dvc init -f
mkdir -p .dvc/local_remote
dvc remote add -d local_remote .dvc/local_remote -f

git add .gitignore .dvc/config dvc.yaml scripts/
git commit -m "init hw5 reproducible pipeline" || true

dvc repro
dvc status
git add dvc.lock
