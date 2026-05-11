---
title: "Building AI Matrix Live"
excerpt: "What happens when you build a public research dashboard with AI, and the dashboard measures how well the world uses AI"
tags: [AI, Open Source, Data Science, Collaboration]
comments: true
modified: 2026-05-11
show_newsletter_signup: true
use_math: false
use_mermaid: false
toc: true
toc_label: Contents
toc_sticky: true
image: /images/placeholder.jpg
header:
  image: /images/placeholder.jpg
  caption: "TODO: header image"
---

## Introduction

Ewan Simpson and I go back to 2009. We overlapped at [KIMEP University](https://www.kimep.kz/) in Almaty, Kazakhstan, where we both covered a variety of roles in the administration. We've stayed in touch across continents and careers since then. When he reached out about building the public-facing infrastructure for his AI Matrix framework, I said yes immediately; partly because the research is genuinely interesting, and partly because it's Ewan.

The result is [AI Matrix Live](https://aimatrixlive.com). It's a dashboard tracking how 115 countries are adopting AI. Ewan wrote [the paper](https://doi.org/10.5281/zenodo.18181372) and handled methodology and editorial; I built the pipeline, the frontend, and the deployment. He [wrote about the project on LinkedIn](TODO). This post is my side of it.

## Access and agency

As Ewan puts it, "AI adoption is usually measured by counting who has the tools. That number is half the story." A country can have widespread AI access and still see its users issuing one-shot commands rather than co-creating. The gap between adoption-as-headline and adoption-as-capability is where the interesting questions sit.

![screenshot-of-the-website](todo)

The AI Matrix separates these two things. Access on one axis, agency on the other, four quadrants. Ewan first laid out the framework in a conceptual paper on SSRN ([Simpson, 2025](https://doi.org/10.2139/ssrn.5228571)), but it lacked empirical grounding because the data didn't exist yet. The [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex) fixed that. Starting in September 2025, Anthropic began publishing country-level data on Claude conversations: collaboration patterns, task success rates, education-level estimates, use-case shares. Three waves are now public. That's enough to actually place countries on the matrix and watch them move. The operationalized version became the [Decomposing the Capability Overhang](https://doi.org/10.5281/zenodo.18181372) paper, and the dashboard is where it lives.

Between September 2025 and March 2026, mean agency scores rose by around 25%, while per-capita access barely moved. The gap between who has AI and who uses it well is closing. The gap between who has AI and who does not is not.

## The collaboration

Ewan is based in Almaty. I'm in Los Angeles. Working across time zones like this is already natural at Delphos Labs, where I manage teammates across four continents. Async handoffs are a basic part of the workday; when I log off, a colleague on another continent can pick up where I left off. The same pattern applied here.

Ewan writes in his LinkedIn piece: "Central Asia and Silicon Valley are not a connection most observers think to draw." It's Los Angeles in my case, but the point stands. I spent four years in Almaty. Ewan has been there much longer. The professional and technical talent there is real; it was real when we were both at KIMEP, and the gap has narrowed considerably since then. The fact that a research dashboard backed by an academic paper got built by two people twelve time zones apart, largely asynchronously, is not remarkable because of the technology. It's remarkable because it would have been unremarkable to either of us.

## The architecture

The system has two parts: a Python data pipeline and a static website. I don't have the time to maintain a complex webapp, but I can glance at a GitHub Action log every once in a while. I've [written before](/blog/2026/02/01/newsletter-automation/) about using GitHub Actions as free compute for small automations; same principle here. Keep it simple, keep it running.

### The pipeline

Every Monday at 08:00 UTC, a [GitHub Actions workflow](https://github.com/michaelquinn32/ai-matrix-live/blob/main/.github/workflows/update.yml) wakes up and runs the pipeline. The script uses the HuggingFace Hub library to list the contents of Anthropic's `EconomicIndex` dataset repository and compares folder names against a stored record of the last processed release. If a new release folder exists, processing continues. If not, the script exits cleanly.

When there's new data, the pipeline downloads the country-level CSV and runs the paper's methodology: extract country-level metrics from the long-format AEI data, compute an agency composite, compute an access score, and assign each country to a stage.

The agency composite is the mean of up to four min-max normalized components:

- the proportion of conversations classified as co-creation: task iteration, learning, and validation combined
- the proportion classified as directive: inverted, so higher means lower agency
- the task success rate
- and the prompt education level

I say "up to four" because different AEI releases include different variables. The composite adapts to however many components are available, which matters because Anthropic has expanded the data they publish across releases.

```python
# From methodology.py — the adaptive agency composite
components: list[str] = []

if "co_creation" in result.columns and result["co_creation"].notna().any():
    result["norm_co_creation"] = _min_max_normalize(result["co_creation"].fillna(0))
    components.append("norm_co_creation")

if "collab_directive" in result.columns:
    result["norm_directive_inv"] = 1.0 - _min_max_normalize(
        result["collab_directive"].fillna(0)
    )
    components.append("norm_directive_inv")

# ... task success, education level follow the same pattern

result["agency_composite"] = result[components].mean(axis=1)
```

The access score is simpler: raw conversation counts divided by World Bank population figures, then log-transformed to compress the right skew. Countries need at least 200 conversations and 500,000 population to be included; microstates produce unreliable per-capita figures when a handful of API calls can spike their usage rates.

Stage assignment follows sequential threshold rules from the paper. Coursework share above 30% assigns Stage 1 (Full Dependency). Work share above 48% with coursework below 25% assigns Stage 2 (Elite Empowerment). Personal use above 38% assigns Stage 3 (Passive Dependency). Countries that don't match any primary rule get classified by their agency composite score using residual cutpoints. The thresholds come directly from the published methodology; none of this is our invention.

The pipeline writes two JSON files: `countries.json` (latest wave only) and `waves.json` (all waves cumulative). If the data changed, the workflow commits the new files and pushes. Cloudflare Pages rebuilds the site automatically. The whole process takes under two minutes. No human touches it.

### The site

The frontend is an [Astro](https://astro.build/) static site with zero framework dependencies at runtime. Astro compiles the pages to static HTML, and all interactivity comes from vanilla JavaScript in inline `<script>` blocks. The only external dependency is [Plotly.js](https://plotly.com/javascript/), loaded from a CDN.

The home page presents the core scatter, with each country plotted on access against agency, sized by population, and tracked across the three AEI waves with Gapminder-style animated playback. The Explore page lets visitors interrogate single countries: their access and agency trajectories, their use-case mix, their collaboration patterns, their task success rates. The Framework page explains the four quadrants. The Methodology page is the most important of the four: it documents data sources, the normalization procedure, the stage assignment rules, the variable availability across waves, and the limitations. Every choice is open to scrutiny, with all code on GitHub.

The background is a slowly rotating wireframe globe rendered on a `<canvas>` element using a custom orthographic projection. The design is dark-theme only, driven by practicality: Plotly charts render against dark backgrounds, and building a full light-mode variant for every chart wasn't worth the time.

## Build economics

Ewan puts it directly: "A serious public-facing analytical dashboard, with cross-wave country comparisons, an interactive map, a methodologically transparent backend, and a published academic foundation, would have been a six-figure consultancy engagement five years ago." We built it for the cost of a domain name.

The infrastructure cost nothing. I already had a Cloudflare account for my personal website, so deploying AI Matrix Live meant pointing a new domain at a new repository; a few minutes of configuration. GitHub Actions is free for public repos. The data is open. Every library we used is open source.

What's more interesting than the dollar cost is the turnaround. The [commit history](https://github.com/michaelquinn32/ai-matrix-live/commits/main/) tells the story. The first commit was the AGENTS.md file, which is the instruction document that AI coding agents read at the start of every session. I've [written before](/blog/2026/02/14/agents-all-the-way-down/) about code being an emergent property of the system you design around it. The AGENTS.md is that system in miniature: project constraints, stop-list, prose standards, verification steps. Every downstream commit inherited those constraints. From there: a methodology module and exploratory notebook, an Astro site scaffold, a data pipeline wired to HuggingFace, an animated scatter chart, prose cleanup, multiple rounds of visual QA, an accessibility audit from a Lighthouse pass, and deployment to Cloudflare Pages. Each commit references a GitHub issue.

I used the same workflow I use at [Delphos Labs](https://delphoslabs.com). Define the task in an issue. Hand it to an agent. Review the output. Commit. Next issue. The agent read the AGENTS.md, understood the project constraints (don't hardcode release names, don't modify the methodology without discussion, kill slop on sight), and produced code that respected them. The exploratory work happened in [Marimo](https://marimo.io/) notebooks rather than Jupyter, because Marimo notebooks are Python files that version-control cleanly and run reproducibly. The site went from an empty repository to a production deployment; a Python data pipeline pulling from HuggingFace, a four-page interactive dashboard with animated charts and a globe, automated weekly updates, and full methodology documentation.

The AEI data itself reports that typical knowledge-work tasks take around eight to eleven times less time with AI assistance than without it. We lived that ratio building this site.

## What it demonstrates

There is an irony worth naming. The site measures how productively countries engage with AI. The site itself was built using the kind of productive AI engagement it tracks. If you ran the AI Matrix methodology on the process of building AI Matrix Live, we would score high on agency: co-creative collaboration patterns, iterative task refinement, high task success rates. We are a data point in our own dataset.

And here's the thing Ewan gets right in his piece. Five years ago, building something like this would have required a funded team: a frontend developer, a data engineer, a designer, hosting budget, project management overhead. The intellectual work; Ewan's framework, the methodology decisions, the editorial judgment about what belongs on each page; none of that got cheaper. That's the hard part, and it still took everything Ewan brought to it. What collapsed was the production cost around that intellectual work. I didn't need to hire anyone. I didn't need to learn a new charting library from scratch. I didn't need to hand-write deployment scripts. The agent did all of that, and it did it well enough that I spent my time on review and design decisions rather than implementation. This is the [reallocation](/blog/2026/04/18/marginal-analysis/) I keep coming back to: implementation is thin-tailed cost with diminishing returns, so let the agent have it. Methodology and editorial judgment are fat-tailed; a wrong threshold or a misleading chart could undermine the entire project. That's where human time belongs.

That shift matters most for people like Ewan. He has seventeen years in Central Asia, deep familiarity with the AEI data, and the methodological judgment to know which thresholds are defensible. What he didn't have was someone to write the Astro site, wire up the Plotly charts, and deploy the GitHub Action. Tyler Cowen calls context ["that which is scarce"](/blog/2025/08/31/model-for-knowledge-work/); Ewan's context was never the bottleneck. The engineering was. I play the same role at Delphos Labs every day: take someone's domain expertise and build the system around it.

The site is live and free at [aimatrixlive.com](https://aimatrixlive.com). The code is on [GitHub](https://github.com/michaelquinn32/ai-matrix-live). The companion papers are linked from the [methodology page](https://aimatrixlive.com/methodology). Comments and critique welcome.

## Tools Used

- [OpenCode](https://opencode.ai/) for agent-driven development
- [Astro](https://astro.build/) for static site generation
- [Plotly.js](https://plotly.com/javascript/) for interactive charts
- [Marimo](https://marimo.io/) for exploratory data analysis
- Python (pandas, huggingface-hub, pycountry) for the data pipeline
- GitHub Actions for CI/CD automation
- Cloudflare Pages for hosting
- [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex) for source data
- World Bank Open Data API for population figures
