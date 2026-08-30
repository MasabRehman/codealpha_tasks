---
name: Professional Fintech Core
colors:
  surface: '#fcf8fa'
  surface-dim: '#dcd9db'
  surface-bright: '#fcf8fa'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f6f3f5'
  surface-container: '#f0edef'
  surface-container-high: '#eae7e9'
  surface-container-highest: '#e4e2e4'
  on-surface: '#1b1b1d'
  on-surface-variant: '#45464d'
  inverse-surface: '#303032'
  inverse-on-surface: '#f3f0f2'
  outline: '#76777d'
  outline-variant: '#c6c6cd'
  surface-tint: '#565e74'
  primary: '#000000'
  on-primary: '#ffffff'
  primary-container: '#131b2e'
  on-primary-container: '#7c839b'
  inverse-primary: '#bec6e0'
  secondary: '#505f76'
  on-secondary: '#ffffff'
  secondary-container: '#d0e1fb'
  on-secondary-container: '#54647a'
  tertiary: '#000000'
  on-tertiary: '#ffffff'
  tertiary-container: '#191c1e'
  on-tertiary-container: '#818486'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dae2fd'
  primary-fixed-dim: '#bec6e0'
  on-primary-fixed: '#131b2e'
  on-primary-fixed-variant: '#3f465c'
  secondary-fixed: '#d3e4fe'
  secondary-fixed-dim: '#b7c8e1'
  on-secondary-fixed: '#0b1c30'
  on-secondary-fixed-variant: '#38485d'
  tertiary-fixed: '#e0e3e5'
  tertiary-fixed-dim: '#c4c7c9'
  on-tertiary-fixed: '#191c1e'
  on-tertiary-fixed-variant: '#444749'
  background: '#fcf8fa'
  on-background: '#1b1b1d'
  surface-variant: '#e4e2e4'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
  data-mono:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  container-max: 1280px
  gutter: 1.5rem
  margin-mobile: 1rem
  margin-desktop: 2rem
  unit-xs: 0.25rem
  unit-sm: 0.5rem
  unit-md: 1rem
  unit-lg: 1.5rem
  unit-xl: 2rem
---

## Brand & Style

The design system is engineered for a high-stakes financial environment where data integrity and user trust are paramount. The brand personality is **Analytical, Authoritative, and Transparent**. It aims to evoke a sense of calm control, transforming complex credit data into actionable insights without unnecessary visual noise.

The design style follows a **Corporate / Modern** aesthetic with a focus on high-legibility and functional density. By utilizing a "Safe" visual identity, the system prioritizes professional clarity through a clean white foundation, rigorous grid alignment, and the strategic use of deep navy to anchor the interface. Every element is designed to feel intentional and precise, catering to users who require rapid, accurate interpretation of financial health.

## Colors

The palette is rooted in a deep navy (#0F172A), which serves as the primary color for navigation, headers, and critical call-to-actions, establishing immediate institutional trust.

- **Primary:** Navy (#0F172A) is used for high-contrast elements and text to ensure maximum readability.
- **Secondary/Neutral:** A scale of Slate Grays provides subtle differentiation for secondary information and structural borders, maintaining a sophisticated, data-centric look.
- **Functional (Risk Tiers):** A specialized semantic palette is used for credit scoring:
    - **Emerald Green:** Represents "Low Risk" / Excellent standing.
    - **Amber:** Represents "Medium Risk" / Action required.
    - **Crimson:** Represents "High Risk" / Warning.
- **Backgrounds:** The interface utilizes a stark white (#FFFFFF) for primary work surfaces with a very light neutral tint (#F8FAFC) for container backgrounds to reduce eye strain during long analytical sessions.

## Typography

This design system exclusively utilizes **Inter** for all typographic needs. Inter was selected for its exceptional legibility at small sizes and its neutral, utilitarian character which suits data-heavy fintech applications.

The type hierarchy is structured to support rapid scanning of financial reports. Headlines use tighter letter-spacing and heavier weights to provide clear section anchors. For data points and tabular information, the system utilizes "Tabular Numerics" (tnum) to ensure that columns of figures align vertically, aiding in quick comparative analysis. Label styles are set in uppercase with increased tracking to distinguish them clearly from body content.

## Layout & Spacing

The layout follows a **Fixed Grid** philosophy on desktop to maintain information density and prevent content from becoming uncomfortably wide for data reading. 

- **Grid System:** A 12-column grid is used for desktop (1280px max-width), transitioning to a 4-column grid for mobile.
- **Spacing Rhythm:** An 8px (0.5rem) base unit governs all dimensions. Gutters are fixed at 24px (1.5rem) to provide enough "air" between dense data containers.
- **Mobile Reflow:** Metric cards and data tables reflow from multi-column layouts to stacked vertical lists or horizontally scrollable containers to preserve data integrity without compromising touch targets.

## Elevation & Depth

Elevation in this design system is achieved through **Low-contrast Outlines** and **Tonal Layers** rather than heavy shadows. This approach keeps the interface flat and professional, avoiding the visual "heaviness" that can distract from data.

1. **Base Surface:** The main background is `#F8FAFC`.
2. **Container Level:** Active work areas (cards, tables) are pure white `#FFFFFF` with a 1px border of `#E2E8F0`.
3. **Interactive Level:** On hover, elements gain a subtle, diffused ambient shadow (0px 4px 6px -1px rgba(15, 23, 42, 0.05)) to indicate interactivity.
4. **Overlay Level:** Modals and dropdowns use a slightly more pronounced shadow and a backdrop blur to maintain focus on the task at hand.

## Shapes

The shape language is **Soft (Level 1)**. This subtle rounding (4px for standard components) balances the clinical precision of the grid with a modern, approachable feel. 

- **Standard Elements:** Buttons, input fields, and small tags use `0.25rem` (4px) corner radii.
- **Large Containers:** Metric cards and dashboard sections use `0.5rem` (8px) to soften the overall layout.
- **Interactive States:** Focus rings are sharp but follow the curvature of the element, maintaining a 2px offset for accessibility.

## Components

- **Buttons:** Primary buttons are solid Navy (#0F172A) with white text. Secondary buttons use a slate border and text. High-risk actions use a crimson outline.
- **High-Density Data Tables:** Optimized for large datasets. Table headers are sticky, using a light slate background and uppercase labels. Row heights are compact (40px) to maximize visible data.
- **Metric Cards:** Large-scale numbers with supporting "Sparklines" (miniature trend charts). These use semantic coloring (green/red) based on the trend direction.
- **Input Fields:** Clean, bordered inputs with 1px slate outlines that thicken and turn Navy on focus. Validation states use the crimson functional color for errors.
- **Risk Badges:** Small, rounded chips that use a high-contrast background (the functional colors) to immediately categorize a user's credit tier.
- **Progress Steppers:** Horizontal indicators for the "Financial Profile" input process, showing completion states with Navy circles and checked icons.