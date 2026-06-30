#!/usr/bin/env python3
"""
Demo script to show different tumor probability outputs
Tests images with varying tumor detection probabilities
"""

import importlib
import os
import io
from pathlib import Path

# Load Flask app
m = importlib.import_module('web.app')
app = m.app

# Test images directory
data_dir = Path('data') / 'images'

# Test images with expected ranges
test_cases = [
    ('3.png', '0% probability (No tumor)'),
    ('10.png', '0% probability (No tumor)'),
    ('100.png', 'High probability (Tumor likely)'),
    ('200.png', 'High probability (Tumor likely)'),
    ('500.png', 'High probability (Tumor likely)'),
]

print("=" * 80)
print("BRAIN TUMOR DETECTION - OUTPUT DEMONSTRATION")
print("=" * 80)

client = app.test_client()

for img_name, description in test_cases:
    img_path = data_dir / img_name
    
    if not img_path.exists():
        print(f"\n[SKIP] {img_name}: File not found")
        continue
    
    print(f"\n{'-' * 80}")
    print(f"Image: {img_name} - {description}")
    print(f"{'-' * 80}")
    
    try:
        with open(img_path, 'rb') as f:
            data = {'file': (io.BytesIO(f.read()), img_name)}
            resp = client.post('/', data=data, content_type='multipart/form-data')
            
            if resp.status_code == 200:
                html = resp.data.decode('utf-8')
                
                # Extract key information from HTML
                if '🧠 Tumor Detected' in html:
                    print("✗ RESULT: 🧠 Tumor Detected")
                elif '✅ No Tumor Detected' in html:
                    print("✓ RESULT: ✅ No Tumor Detected")
                
                # Extract confidence score
                import re
                conf_match = re.search(r'Tumor Confidence: (\d+)%', html)
                if conf_match:
                    confidence = conf_match.group(1)
                    print(f"  Confidence: {confidence}%")
                
                # Extract affected area
                affected_match = re.search(r'<div class="affected-area[^>]*>\s*(.*?)\s*</div>', html, re.DOTALL)
                if affected_match:
                    affected = affected_match.group(1).strip()
                    print(f"  Description: {affected[:100]}...")
                
                # Extract image sources
                original_match = re.search(r'uploads/([^\s"]+)', html)
                if original_match:
                    print(f"  Images saved: {original_match.group(1)} and pred_{original_match.group(1)}")
                
                print("  ✓ Successfully processed")
            else:
                print(f"  ✗ Error: HTTP {resp.status_code}")
                
    except Exception as e:
        print(f"  ✗ Exception: {str(e)}")

print("\n" + "=" * 80)
print("DEMONSTRATION COMPLETE")
print("=" * 80)
print("\nOUTPUT TYPES:")
print("  • 0% Probability: ✅ No Tumor Detected (Green background)")
print("  • 50%+ Probability: 🧠 Tumor Detected (Red background)")
print("  • All results show: Confidence %, Affected Area, Original + Predicted Images")
print("=" * 80)
