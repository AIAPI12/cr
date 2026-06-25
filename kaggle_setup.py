#!/usr/bin/env python3
"""
Kaggle one-shot setup + training launcher.
Run this as a Kaggle notebook cell:
    !python kaggle_setup.py --games 250000 --step-n 400

Handles: gamedata.json, Oracle SSH key, checkpoint sync, training.
"""
import os, sys, subprocess, shutil, json

PROJ = '/kaggle/working/cr'
os.makedirs(PROJ, exist_ok=True)
if os.getcwd() != PROJ:
    os.chdir(PROJ)

# Ensure gamedata.json exists
if not os.path.exists('gamedata.json'):
    candidates = [
        '/kaggle/input/cr-data/gamedata.json',
        '/kaggle/working/gamedata.json',
    ]
    for p in candidates:
        if os.path.exists(p):
            shutil.copy2(p, 'gamedata.json')
            break
    if not os.path.exists('gamedata.json'):
        print('WARNING: gamedata.json not found! Training will crash.')
        print('Upload it as Kaggle Dataset "cr-data" or place in /kaggle/working/')

# Set up Oracle SSH key from Kaggle secret
ssh_key_path = '/kaggle/working/oracle_key'
ssh_key = os.environ.get('ORACLE_SSH_KEY', '')
if ssh_key and not os.path.exists(ssh_key_path):
    import base64
    try:
        key_bytes = base64.b64decode(ssh_key).decode('utf-8')
        with open(ssh_key_path, 'w') as f:
            f.write(key_bytes)
        os.chmod(ssh_key_path, 0o600)
        os.environ['ORACLE_SSH_KEY'] = ssh_key_path
    except Exception:
        os.environ['ORACLE_SSH_KEY'] = ssh_key

print('Starting training...')
sys.stdout.flush()

cmd = [sys.executable, 'kaggle_train.py'] + sys.argv[1:]
os.execvp(sys.executable, cmd)
