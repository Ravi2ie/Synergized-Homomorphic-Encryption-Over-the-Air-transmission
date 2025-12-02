#!/usr/bin/env python3
"""
Create placeholder GIFs for attack visualizations
Run this script once to generate all 5 GIFs
"""

import os
from PIL import Image, ImageDraw, ImageFont
import imageio

# Create gifs directory if it doesn't exist
os.makedirs('gifs', exist_ok=True)

# Define attacks with colors
attacks = {
    'traffic_analysis.gif': {
        'title': 'Traffic Analysis',
        'color': '#FF6B6B',
        'subtitle': 'Observing Communication Patterns'
    },
    'tampering.gif': {
        'title': 'Tampering',
        'color': '#4ECDC4',
        'subtitle': 'Data Modification Detected'
    },
    'differential_inference.gif': {
        'title': 'Differential Inference',
        'color': '#45B7D1',
        'subtitle': 'Privacy Breach Through Analysis'
    },
    'insider_compromise.gif': {
        'title': 'Insider Compromise',
        'color': '#FFA07A',
        'subtitle': 'System Access Compromised'
    },
    'dos.gif': {
        'title': 'Denial of Service',
        'color': '#FFD700',
        'subtitle': 'Server Overload Attack'
    }
}

print("🎬 Creating Attack Visualization GIFs...\n")

for filename, info in attacks.items():
    frames = []
    
    # Create 15 frames for smooth animation
    for frame_num in range(15):
        # Create base image
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw colored background
        intensity = 0.5 + 0.5 * (frame_num / 15)  # Fade in/out effect
        color_hex = info['color']
        
        # Calculate position for animated box
        box_size = int(50 + 30 * abs(frame_num - 7) / 7)
        x = (400 - box_size) // 2
        y = 80
        
        # Draw colored rectangle
        draw.rectangle(
            [(x, y), (x + box_size, y + box_size)],
            fill=color_hex
        )
        
        # Draw title
        try:
            draw.text((20, 150), info['title'], fill='#333333', font=None)
        except:
            draw.text((20, 150), info['title'], fill='#333333')
        
        # Draw subtitle
        try:
            draw.text((20, 180), info['subtitle'], fill='#666666', font=None)
        except:
            draw.text((20, 180), info['subtitle'], fill='#666666')
        
        # Draw frame counter
        draw.text((350, 10), f"{frame_num+1}/15", fill='#999999')
        
        frames.append(img)
    
    # Save as GIF
    gif_path = os.path.join('gifs', filename)
    imageio.mimsave(gif_path, frames, duration=0.1, loop=0)
    
    print(f"✅ Created {filename}")
    print(f"   Title: {info['title']}")
    print(f"   Path: {gif_path}")
    print(f"   Size: {os.path.getsize(gif_path) / 1024:.1f} KB\n")

print("=" * 50)
print("✅ All GIFs created successfully!")
print("=" * 50)
print("\nGIF files location:")
print(f"  {os.path.abspath('gifs')}\n")
print("Files created:")
for filename in attacks.keys():
    gif_path = os.path.join('gifs', filename)
    print(f"  - {filename}: {os.path.getsize(gif_path) / 1024:.1f} KB")
