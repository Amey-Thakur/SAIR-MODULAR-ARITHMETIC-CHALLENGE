# Branding and Visual Identity Plan

## 1. Overview
The SAIR Modular Arithmetic Challenge repository must visually communicate "Mathematics," "Deep Learning," and "Professional Research Laboratory." Borrowing from the standard set by the Mathematics Distillation repository, all assets will adhere to a consistent color palette, typographic structure, and geometric style.

## 2. Color Palette
- **Primary Accent (SAIR Blue)**: `#0D47A1` - Represents formal logic and trust.
- **Secondary Accent (Neural Purple)**: `#4A148C` - Represents deep learning and nonlinear representations.
- **Background/Base**: `#0F172A` (Slate 900) - For dark mode diagrams.
- **Text/Line Art**: `#F1F5F9` (Slate 100) or `#1E293B` (Slate 800) depending on background.

## 3. Typography
- **Headings**: Inter or Roboto (Clean, sans-serif, modern).
- **Monospace/Code/Math**: JetBrains Mono or Fira Code (Ligatures for mathematical operators are essential).

## 4. Logo Concept
The logo should encapsulate modular arithmetic and neural networks.
- **Concept**: A cyclic graph (representing modulo $p$) intersecting with a multi-layered perceptron (nodes and edges).
- **Execution**: Scalable Vector Graphic (SVG) ensuring crisp resolution across all devices.
- **Mathematical Significance**: The nodes of the cyclic graph should be $p$ nodes (abstracted), and the routing between them represents the multiplication operation wrapping around the modulus.

## 5. Asset Hierarchy
- `branding/logos/sair_modular_logo_dark.svg`
- `branding/logos/sair_modular_logo_light.svg`
- `branding/social_preview.png`: 1280x640 image for GitHub link previews. Should feature the logo, the challenge title, and a subtle background of transformer attention maps.

## 6. Architecture Diagrams
All technical diagrams (e.g., tokenization schemes, routing architectures) will be created using:
- **Mermaid JS** for inline Markdown rendering.
- **SVG exports** (via tools like Excalidraw or Draw.io) for complex, high-fidelity neural network schematics.

## 7. Badges
The README will feature professional shields.io badges:
- Build Status
- License (MIT)
- Hugging Face Model Hub (Placeholder)
- Python 3.10+
- PyTorch 2.0+

## 8. Implementation Steps
1. Generate the base SVG logo using mathematical plotting (e.g., matplotlib to generate points, styled in SVG).
2. Create the `social_preview.png` composite.
3. Establish Mermaid templates for our model architectures.
