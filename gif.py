from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math
import subprocess

# Enhanced element coordinates with better spacing
elements = {
    "Driver 1": (80, 100),
    "Driver 2": (80, 200),
    "Driver 3": (80, 300),
    "Server": (350, 200),
    "Data Analyst": (620, 200)
}

# Arrows (flow) with professional styling
arrows = [
    ("Driver 1", "Server", "🔒"),
    ("Driver 2", "Server", "🔒"),
    ("Driver 3", "Server", "🔒"),
    ("Server", "Data Analyst", "⚡")
]

# Enhanced labels with better descriptions
labels = {
    "Driver 1": ("Driver", "Encrypt earnings"),
    "Driver 2": ("Driver", "Encrypt earnings"), 
    "Driver 3": ("Driver", "Encrypt earnings"),
    "Server": ("Server", "Receives & decrypts data"),
    "Data Analyst": ("Analyst", "Views aggregated results")
}

def calculate_arrow_polygon(x1, y1, x2, y2, arrowhead_height=10, tail_thickness=4):
    """Calculate polygon points for a professional arrow"""
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return []
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    
    base_cx = x2 - arrowhead_height * ux
    base_cy = y2 - arrowhead_height * uy
    tri_base_half_width = arrowhead_height / math.sqrt(3)
    tail_half = tail_thickness / 2
    
    # Calculate the 7 points of the arrow polygon
    p1 = (x1 + px * tail_half, y1 + py * tail_half)
    p2 = (x1 - px * tail_half, y1 - py * tail_half)
    p3 = (base_cx - px * tail_half, base_cy - py * tail_half)
    p7 = (base_cx + px * tail_half, base_cy + py * tail_half)
    p4 = (base_cx - px * tri_base_half_width, base_cy - py * tri_base_half_width)
    p6 = (base_cx + px * tri_base_half_width, base_cy + py * tri_base_half_width)
    p5 = (x2, y2)
    
    return [p1, p2, p3, p4, p5, p6, p7]

def create_professional_frame(elements, arrows, labels, current_step, frame_size=(800, 500)):
    """Create a single frame with enhanced visuals"""
    img = Image.new("RGB", frame_size, "white")
    draw = ImageDraw.Draw(img)
    
    # Try to load better fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 14)
        desc_font = ImageFont.truetype("arial.ttf", 10)
        emoji_font = ImageFont.truetype("seguiemj.ttf", 24)
    except:
        # Fallback to default fonts
        title_font = ImageFont.load_default()
        desc_font = ImageFont.load_default()
        emoji_font = ImageFont.load_default()
    
    # Draw background grid (subtle)
    for x in range(0, frame_size[0], 50):
        draw.line([(x, 0), (x, frame_size[1])], fill="#f5f5f5", width=1)
    for y in range(0, frame_size[1], 50):
        draw.line([(0, y), (frame_size[0], y)], fill="#f5f5f5", width=1)
    
    # Draw elements with rounded rectangles and gradients
    for name, (x, y) in elements.items():
        title, desc = labels[name]
        
        # Draw rounded rectangle container with shadow effect
        shadow_offset = 2
        draw.rounded_rectangle([
            x-45+shadow_offset, y-45+shadow_offset, 
            x+145+shadow_offset, y+45+shadow_offset
        ], radius=12, fill="#e0e0e0", outline="#e0e0e0")
        
        # Main container
        draw.rounded_rectangle([x-45, y-45, x+145, y+45], 
                              radius=12, fill="#ffffff", outline="#505050", width=2)
        
        # Draw emoji and text with better fonts
        if "Driver" in name:
            emoji = "🚗"
            color = "#2E86AB"
        elif "Server" in name:
            emoji = "🖥️" 
            color = "#A23B72"
        else:
            emoji = "👨‍💻"
            color = "#F18F01"
            
        draw.text((x-35, y-30), emoji, font=emoji_font, fill="black")
        draw.text((x, y-25), name, font=title_font, fill=color)
        draw.text((x, y+0), f"{title}", font=desc_font, fill="#404040")
        draw.text((x, y+12), f"{desc}", font=desc_font, fill="#666666")
    
    # Draw animated arrows
    for i, (start, end, emoji) in enumerate(arrows):
        if i >= current_step:
            continue
            
        x1, y1 = elements[start]
        x2, y2 = elements[end]
        
        # Adjust coordinates to start/end at element edges
        x1_adj, y1_adj = x1 + 145, y1
        x2_adj, y2_adj = x2 - 45, y2
        
        # Calculate arrow color based on type
        arrow_color = "#0066cc" if "🔒" in emoji else "#ff6600"
        
        # Draw arrow line
        if i == current_step - 1:
            # Animate the current arrow (partial growth)
            progress = min(1.0, (current_step) / len(arrows))
            x2_current = x1_adj + (x2_adj - x1_adj) * progress
            y2_current = y1_adj + (y2_adj - y1_adj) * progress
        else:
            # Draw completed arrow
            x2_current, y2_current = x2_adj, y2_adj
        
        # Draw arrow line
        draw.line([x1_adj, y1_adj, x2_current, y2_current], 
                 fill=arrow_color, width=3)
        
        # Draw arrowhead for completed arrows
        if i < current_step - 1:
            arrow_points = calculate_arrow_polygon(
                x1_adj, y1_adj, x2_adj, y2_adj, 
                arrowhead_height=12, tail_thickness=4
            )
            if arrow_points:
                draw.polygon(arrow_points, fill=arrow_color)
        
        # Draw emoji label at midpoint
        mx, my = (x1_adj + x2_current) // 2, (y1_adj + y2_current) // 2
        # Add background for emoji
        draw.rectangle([mx-15, my-15, mx+15, my+15], fill="white", outline=arrow_color, width=1)
        draw.text((mx-8, my-12), emoji, font=emoji_font, fill="black")
    
    # Add step counter
    draw.text((10, 10), f"Step {current_step}/{len(arrows)}", font=title_font, fill="#666666")
    
    return img

# Main execution
print("🚀 Creating professional animated GIF...")

frames = []
total_steps = len(arrows) + 1

# Create frames with smooth progression
for step in range(total_steps):
    print(f"🔄 Generating frame {step + 1}/{total_steps}")
    frame = create_professional_frame(elements, arrows, labels, step)
    frames.append(frame)

# Save with optimization
print("💾 Saving GIF file...")
frames[0].save("professional_driver_flow.gif", 
               save_all=True, 
               append_images=frames[1:], 
               duration=1000,  # 1 second per frame
               loop=0, 
               optimize=True, 
               palette=Image.ADAPTIVE, 
               colors=128)

print("✅ Professional GIF saved as 'professional_driver_flow.gif'")

# Try to further optimize with Gifsicle if available
try:
    print("🛠️  Applying additional optimization with Gifsicle...")
    subprocess.run([
        "gifsicle", 
        "--optimize=3", 
        "--lossy=30", 
        "-o", "optimized_driver_flow.gif", 
        "professional_driver_flow.gif"
    ], check=True)
    print("✅ Further optimized version saved as 'optimized_driver_flow.gif'")
except FileNotFoundError:
    print("ℹ️  Gifsicle not found. Install for additional compression:")
    print("   Windows: Download from https://eternallybored.org/misc/gifsicle/")
    print("   Mac: brew install gifsicle")
    print("   Linux: sudo apt-get install gifsicle")
except Exception as e:
    print(f"⚠️  Gifsicle optimization failed: {e}")

print("🎉 Animation creation completed!")