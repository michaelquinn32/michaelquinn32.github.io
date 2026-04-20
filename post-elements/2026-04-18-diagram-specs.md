# Diagram Specifications for "Everything has a cost. Everything has a benefit."

Five diagrams, in priority order. All should share a consistent visual style: clean, minimal, no decoration. White or very light background. Sans-serif font (Inter, Helvetica, or similar). No drop shadows, no gradients, no 3D effects. The blog uses Minimal Mistakes theme; diagrams should feel like they belong in a technical article, not a slide deck.

Color palette: use 4 colors consistently across all diagrams.
- **Green** (#2E7D32 or similar): automate / thin-tailed / low human investment
- **Red** (#C62828 or similar): maximum human investment / fat-tailed / high stakes
- **Amber** (#F9A825 or similar): mixed / delegate with oversight
- **Blue** (#1565C0 or similar): neutral / labels / axes

---

## Diagram 1: The 2x2 Framework

**Filename**: `2x2-framework.png`
**Placement**: Section II, replacing or supplementing the markdown table
**Purpose**: The central visual of the article. Must be immediately legible and memorable.

**Layout**: A 2x3 matrix (2 columns × 3 rows), with labeled axes.

- **X-axis** (columns): "Thin-tailed cost" | "Fat-tailed cost"
- **Y-axis** (rows, top to bottom): "Sub-linear benefit" | "Linear benefit" | "Super-linear benefit"

**Cell contents** (each cell gets a label and a color fill):

| | Thin-tailed cost | Fat-tailed cost |
|--|--|--|
| Sub-linear benefit | AUTOMATE FULLY (green fill) | GATE AND MONITOR (amber fill) |
| Linear benefit | DELEGATE + SPOT CHECK (light green fill) | INVEST PROPORTIONALLY (light amber fill) |
| Super-linear benefit | AI-ASSISTED, HUMAN-LED (amber fill) | MAXIMUM HUMAN INVESTMENT (red fill) |

**Additional element**: A large curved arrow from top-left cell to bottom-right cell, labeled "Reallocate human time." The arrow should be prominent but not obscure the cells. Dashed line, dark gray or blue.

**Dimensions**: ~800×500px. Landscape orientation.

**Caption**: "The marginal analysis framework. Classify each SDLC phase by cost distribution and benefit scaling. The arrow shows the reallocation direction that abundance demands."

---

## Diagram 2: SDLC Phases Plotted on the 2x2

**Filename**: `sdlc-phases-mapped.png`
**Placement**: Between Section III (Inner Loop) and Section IV (Outer Loop), or at the end of Section IV as a summary
**Purpose**: Makes the framework concrete by showing where each phase lands.

**Layout**: Same 2x3 grid as Diagram 1, but cells are lightly shaded (no text labels in cells). Instead, SDLC phases are plotted as labeled dots on the grid.

**Phase placements** (approximate positions within the grid):

Top-left region (thin-tailed, sub-linear):
- "Style review" (green dot)
- "Formatting" (green dot)
- "Boilerplate" (green dot)

Top-right region (fat-tailed, sub-linear):
- (empty; nothing fits here well)

Middle-left (thin-tailed, linear):
- "Routine bug fixes" (light green dot)
- "Implementation (commodity)" (light green dot)

Middle-right (fat-tailed, linear):
- "Per-service monitoring" (amber dot)
- "Dependency updates" (amber dot)

Bottom-left (thin-tailed, super-linear):
- "TDD / test writing" (amber dot; positioned near the center line because it's thin-tailed to produce but mitigates fat-tail risks)

Bottom-right (fat-tailed, super-linear):
- "Task definition" (red dot, large)
- "Security threat modeling" (red dot, large)
- "Core architecture" (red dot, large)
- "Product strategy" (red dot, large)
- "Observability design" (red dot)
- "Shots on goal / PMF" (red dot)

**TDD special treatment**: Draw a dashed arrow from TDD's position toward the bottom-right, labeled "mitigates fat-tail risks." This captures the dual nature: cheap to produce, but its value is in catching fat-tailed problems.

**Dimensions**: ~900×600px. Landscape.

**Caption**: "Every SDLC phase classified. Green dots are fully automatable. Red dots demand maximum human judgment. TDD is cheap to produce but catches fat-tailed risks, earning its place near the center."

---

## Diagram 3: Cost Distribution Curves

**Filename**: `cost-distributions.png`
**Placement**: Section II, near the "Axis 1: How costs distribute" subsection
**Purpose**: Grounds the thin-tail / fat-tail distinction visually. Many readers will grasp the concept faster from the curves than from prose.

**Layout**: A single chart with two overlapping probability distributions plotted on the same x-axis.

- **X-axis**: "Cost magnitude (human hours)" — left side is low cost, right side is high cost
- **Y-axis**: "Frequency"

**Curve 1 — Thin-tailed** (green, solid line):
- A normal distribution, tight and centered. Peaks high, falls off quickly on both sides.
- Label: "Thin-tailed costs (formatting, routine bugs, boilerplate)"
- An annotation arrow pointing to the right tail: "Bounded downside"

**Curve 2 — Fat-tailed** (red, solid line):
- A power-law / Pareto distribution. Lower peak, but a long right tail extending far to the right.
- Label: "Fat-tailed costs (security breaches, wrong architecture, bad strategy)"
- An annotation arrow pointing to the far right tail: "Unbounded downside"
- The area under the far right tail (past some threshold) should be lightly shaded in red to emphasize the catastrophic tail.

**Key visual contrast**: The green curve should be obviously contained; the red curve should obviously extend far beyond it on the right side. The message is: "you can't manage these the same way."

**Dimensions**: ~700×400px. Landscape.

**Caption**: "Thin-tailed costs shrink toward zero with AI. Fat-tailed costs have unbounded downside and don't shrink with abundance."

---

## Diagram 4: The Inner Loop with Quadrant Classification

**Filename**: `inner-loop-classified.png`
**Placement**: Section III, at the top before the subsections begin
**Purpose**: A quick visual map of the inner loop phases and their classification.

**Layout**: A horizontal flow, left to right, with 5 boxes connected by arrows:

```
[Task Definition] → [TDD] → [Implementation] → [Code Review] → [PR / Commit]
```

Each box is color-coded by its quadrant classification:
- **Task Definition**: Red fill, white text. Label below: "Fat-tail × Super-linear"
- **TDD**: Amber fill, dark text. Label below: "Thin cost, fat-tail mitigator"
- **Implementation**: Green fill, dark text. Label below: "Thin-tail × Sub-linear"
- **Code Review**: Split into three horizontal stripes within the box:
  - Top stripe (green): "Style"
  - Middle stripe (red): "Security"
  - Bottom stripe (amber): "Architecture"
  - Label below: "Mixed"
- **PR / Commit**: Amber fill, dark text. Label below: "Restructure"

**Below the flow**: A horizontal bar or gradient showing "Human attention investment" — low on the right (implementation), high on the left (task definition) and in the security stripe of code review. This reinforces that attention should concentrate at the bookends and on security.

**Dimensions**: ~900×350px. Landscape.

**Caption**: "The inner loop, classified. Human attention concentrates at task definition and security review. Everything in between is increasingly automated."

---

## Diagram 5: The Jevons Feedback Loop

**Filename**: `jevons-loop.png`
**Placement**: Section V, near the "Jevons across both loops" subsection
**Purpose**: Shows why abundance doesn't free you; it shifts the constraint.

**Layout**: A circular flow with 5 nodes, arranged clockwise:

1. **"AI makes software cheaper"** (green box, top)
2. **"More software gets built"** (blue box, right) — with a small annotation: "GitHub pushes +35% YoY, iOS apps +50%"
3. **"Attack surface grows / Maintenance burden grows / More judgment calls"** (red box, bottom) — this is the widest box, three lines of text
4. **"Human attention becomes the bottleneck"** (red box, left)
5. **"Demand for human judgment increases"** (amber box, top-left)

Arrows connect each node to the next, forming a circle. The arrow from node 5 back to node 1 should be dashed, with a small label: "cycle repeats."

**Center of the circle**: Text in italic: "The Jevons Paradox for software"

**Dimensions**: ~600×600px. Square.

**Caption**: "Cheaper software doesn't free human attention. It shifts demand to judgment, monitoring, and security; the fat-tailed work that abundance creates."

---

## General Notes

- All diagrams should be saved as PNG at 2x resolution (so a 900×600 diagram is actually 1800×1200px) for retina displays.
- Alt text for each diagram should match the caption.
- File names use kebab-case and live in `/images/2026-04-18/`.
- If generating with code (matplotlib, d3, etc.), keep the style minimal. No chartjunk. Tufte principles: maximize data-ink ratio.
