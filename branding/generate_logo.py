#!/usr/bin/env python3

"""
Generate the SAIR Modular Arithmetic Challenge social preview using raw mathematical computation.
Instead of AI-generated art, this uses the exact style of the Mathematics Distillation repository:
representing the underlying problem domain mathematically as a pixel grid.

We visualize the quadratic residues and modular multiplication over a prime field.
"""

from PIL import Image, ImageDraw, ImageFont
import math

# Branding Colors
SLATE_900 = (15, 23, 42)
SAIR_BLUE = (13, 71, 161)
NEURAL_PURPLE = (74, 20, 140)
SLATE_100 = (241, 245, 249)

WIDTH = 1280
HEIGHT = 640
PRIME = 641 # A nice prime number close to the height

def interpolate_color(c1, c2, t):
    """Linear interpolation between two RGB colors."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t)
    )

def generate_banner(output_path):
    print("Generating mathematical banner...")
    img = Image.new("RGB", (WIDTH, HEIGHT), SLATE_900)
    pixels = img.load()

    # Create a mathematical background: Modular Multiplication Table
    # We will map (x, y) to (A, B) and color based on (A * B) % P
    for x in range(WIDTH):
        for y in range(HEIGHT):
            # Scale coordinates slightly for visual density
            A = (x // 2) % PRIME
            B = (y // 2) % PRIME
            
            val = (A * B) % PRIME
            
            # Normalize val to [0, 1]
            t = val / PRIME
            
            # Color mapping: 
            # 0.0 -> SLATE_900 (Background)
            # 0.5 -> SAIR_BLUE
            # 1.0 -> NEURAL_PURPLE
            
            if t < 0.5:
                # Interpolate between SLATE_900 and SAIR_BLUE
                color = interpolate_color(SLATE_900, SAIR_BLUE, t * 2)
            else:
                # Interpolate between SAIR_BLUE and NEURAL_PURPLE
                color = interpolate_color(SAIR_BLUE, NEURAL_PURPLE, (t - 0.5) * 2)
                
            pixels[x, y] = color

    # Overlay geometric abstraction of a Neural Network (Grokking representation)
    draw = ImageDraw.Draw(img)
    
    # Draw some "hidden layers"
    layer_x = [WIDTH // 4, WIDTH // 2, 3 * WIDTH // 4]
    num_nodes = 5
    
    nodes = []
    for lx in layer_x:
        layer_nodes = []
        spacing = HEIGHT // (num_nodes + 1)
        for i in range(1, num_nodes + 1):
            ly = i * spacing
            layer_nodes.append((lx, ly))
        nodes.append(layer_nodes)
        
    # Draw edges between layers
    for i in range(len(nodes) - 1):
        for n1 in nodes[i]:
            for n2 in nodes[i+1]:
                # Draw edges with low opacity/subtle color
                draw.line([n1, n2], fill=(241, 245, 249, 50), width=1)
                
    # Draw nodes
    for layer in nodes:
        for (nx, ny) in layer:
            r = 6
            draw.ellipse([nx-r, ny-r, nx+r, ny+r], fill=SLATE_100)

    # Save the result
    img.save(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_banner("social_preview.png")
