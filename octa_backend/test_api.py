#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json

# Test weight list API
try:
    response = requests.get('http://127.0.0.1:8000/api/v1/weight/list')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
except Exception as e:
    print(f"Error: {e}")
