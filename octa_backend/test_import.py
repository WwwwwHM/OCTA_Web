#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import traceback

try:
    from main import app
    print("✓ main.py import successful")
except Exception as e:
    print(f"✗ Error importing main: {e}")
    traceback.print_exc()
    sys.exit(1)
