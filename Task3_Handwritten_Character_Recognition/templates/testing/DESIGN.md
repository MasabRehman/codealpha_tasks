---
name: Technical Blueprint HCR
colors:
  surface: '#131315'
  surface-dim: '#131315'
  surface-bright: '#39393b'
  surface-container-lowest: '#0e0e10'
  surface-container-low: '#1b1b1d'
  surface-container: '#1f1f21'
  surface-container-high: '#2a2a2b'
  surface-container-highest: '#353436'
  on-surface: '#e4e2e3'
  on-surface-variant: '#c6c6cd'
  inverse-surface: '#e4e2e3'
  inverse-on-surface: '#303032'
  outline: '#909097'
  outline-variant: '#45464c'
  surface-tint: '#bec6e0'
  primary: '#bec6e0'
  on-primary: '#283044'
  primary-container: '#565e74'
  on-primary-container: '#d0d8f2'
  inverse-primary: '#565e74'
  secondary: '#bec6e0'
  on-secondary: '#283044'
  secondary-container: '#3e465c'
  on-secondary-container: '#adb5ce'
  tertiary: '#dbc39b'
  on-tertiary: '#3d2e11'
  tertiary-container: '#6e5c3b'
  on-tertiary-container: '#eed6ac'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#dae2fc'
  primary-fixed-dim: '#bec6e0'
  on-primary-fixed: '#131b2e'
  on-primary-fixed-variant: '#3e465b'
  secondary-fixed: '#dae2fd'
  secondary-fixed-dim: '#bec6e0'
  on-secondary-fixed: '#131b2e'
  on-secondary-fixed-variant: '#3e465c'
  tertiary-fixed: '#f8dfb5'
  tertiary-fixed-dim: '#dbc39b'
  on-tertiary-fixed: '#261901'
  on-tertiary-fixed-variant: '#554425'
  background: '#131315'
  on-background: '#e4e2e3'
  surface-variant: '#353436'
  terminal-bg: '#0b0e14'
  stroke-active: '#ffffff'
  bounding-box: '#7c839b'
  confidence-high: '#4ade80'
  confidence-low: '#f87171'
  grid-line: '#1e293b'
typography:
  headline-xl:
    fontFamily: Space Grotesk
    fontSize: 40px
    fontWeight: '700'
    lineHeight: 48px
    letterSpacing: -0.03em
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
  body-lg:
    fontFamily: JetBrains Mono
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.1em
  data-terminal:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
spacing:
  canvas-margin: 2rem
  gutter-technical: 1px
  sidebar-width: 320px
  stroke-weight-thin: 2px
  stroke-weight-bold: 8px
  unit-base: 4px
---

## Brand & Style

This design system pivots the established professional aesthetic into a **Brutalist / Experimental** technical direction. It is engineered for researchers and developers working with Handwritten Character Recognition (HCR) and Convolutional Neural Networks (CNN). The visual identity moves away from corporate softness toward a "blueprint" aesthetic, emphasizing the algorithmic nature of image processing.

The personality is **Precise, Raw, and Computational**. By utilizing sharp corners, visible grid lines, and monospaced data readouts, the UI evokes the feeling of a diagnostic terminal. It treats the canvas and character analysis as technical specimens, providing a high-contrast environment where stroke analysis and classification confidence are the primary focal points.

## Colors

The palette is optimized for a **Dark Mode** environment to reduce eye strain during high-precision drawing and analysis tasks. The core seed color (#565e74) is expanded into a technical slate spectrum.

- **Primary:** A muted slate navy used for structural elements and UI framing.
- **Secondary:** A bright, cool-toned gray for high-visibility interactive elements.
- **Backgrounds:** A near-black terminal gray (#0B0E14) provides the foundation for the drawing canvas.
- **Functional Overlays:**
    - **Stroke Active:** Pure white for maximum contrast during character input.
    - **Bounding Box:** A semi-transparent slate used to encapsulate identified character regions.
    - **Classification Spectrum:** A neon-tinted green and red system used exclusively for model confidence percentages.

## Typography

The typography leverages a dual-font system to reinforce the technical narrative. **Space Grotesk** is used for headlines, providing a futuristic, geometric feel that anchors the page. **JetBrains Mono** is utilized for all body text, labels, and data outputs to simulate a developer environment and ensure tabular data (like CNN weight matrices) aligns perfectly.

Hierarchy is strictly enforced through weight and letter spacing. Labels are consistently set in uppercase with wide tracking to differentiate them from functional data readouts. Large numerical displays for classification confidence utilize the monospaced font to prevent "jumping" during real-time inference updates.

## Layout & Spacing

The layout utilizes a **Technical Grid** approach where borders are often used as gutters (1px) to create a tiled, "paneled" look. 

- **Drawing Canvas:** Centrally aligned with a fixed aspect ratio (usually 1:1 for MNIST/EMNIST compatibility), surrounded by a diagnostic margin.
- **Sidebar Analytics:** A fixed 320px right-hand panel contains real-time classification results, probability distributions, and stroke analysis.
- **Spacing Rhythm:** Based on a 4px unit. Components are packed tightly to maximize information density, reflecting a professional diagnostic tool rather than a consumer app.
- **Breakpoints:** Desktop (1280px+) shows all panels; Tablet collapses the sidebar into a bottom-sheet; Mobile focuses exclusively on the canvas with an overlay for results.

## Elevation & Depth

In alignment with the Brutalist direction, this design system rejects shadows in favor of **Bold Borders** and **Tonal Layering**.

- **Depth through Borders:** Hierarchy is established by border weight and color. The main canvas has a thicker, high-contrast border, while secondary utility panels use lower-contrast slate lines.
- **Blueprint Grid:** The primary drawing surface features a persistent 10% opacity grid background, referencing coordinate geometry.
- **Active State:** Elements move "forward" not through elevation, but by changing border color from slate to white or by filling the background with a primary slate tint.

## Shapes

The shape language is strictly **Sharp (Level 0)**. Every UI element—from buttons and input fields to the drawing canvas itself—features 0px corner radii. This reinforces the "unrefined" brutalist aesthetic and emphasizes the pixel-perfect nature of image processing and character segmentation. Hard corners allow for seamless tiling of UI panels, creating a cohesive, monolithic interface.

## Components

- **Drawing Canvas:** A specialized component with a dark terminal background and a subtle coordinate grid. It supports variable stroke weights and provides a "clear" button that triggers a CRT-style wipe animation.
- **Bounding Boxes:** Sharp-edged, non-filled rectangles that appear over processed characters. They include a small corner-aligned label showing the predicted character and confidence % (e.g., "A: 98.2%").
- **Classification Chips:** Monospaced tags used in lists to show top-N predictions. They feature a horizontal "confidence bar" background that fills based on the probability value.
- **Technical Buttons:** Rectangular, sharp-edged buttons with high-contrast borders. Inactive states are ghost-style (border only), while primary actions are solid slate.
- **Data Readouts:** Small, bordered boxes containing raw CNN layer outputs or metadata (e.g., "Input: 28x28px", "Model: EMNIST-v2").
- **Stroke Controls:** A specialized toolbar for selecting brush thickness (thin/bold) and input mode (Draw/Erase), using high-contrast icons and sharp toggle states.