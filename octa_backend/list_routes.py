#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json

# List all registered routes
try:
    response = requests.get('http://127.0.0.1:8000/openapi.json')
    data = response.json()
    
    print("=" * 80)
    print("REGISTERED ROUTES")
    print("=" * 80)
    
    for path, methods in data.get('paths', {}).items():
        for method in methods:
            method_upper = method.upper()
            print(f"{method_upper:8} {path}")
    
    print("\n" + "=" * 80)
    print(f"Total routes: {sum(len(m) for m in data.get('paths', {}).values())}")
    print("=" * 80)
    
except Exception as e:
    print(f"Error: {e}")
