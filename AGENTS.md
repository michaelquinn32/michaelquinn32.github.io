# Agent Instructions for michaelquinn32.github.io

This file provides context for AI agents working on this Jekyll blog.

## Site Overview

- **Platform**: Jekyll static site hosted on GitHub Pages
- **Theme**: Minimal Mistakes (remote theme)
- **Domain**: msquinn.com
- **Author**: Michael Quinn
- **Focus**: Data Science, AI/ML, agentic development, and knowledge work

## Blog Post Style Guide

### Structure

Posts follow a consistent structure:

1. **Frontmatter** with title, excerpt, tags, image, header, and feature flags
2. **Introduction** section (`## Introduction`) that frames the problem or topic
3. **Body sections** with descriptive headings
4. **Conclusion** summarizing takeaways
5. **Tools Used** section listing AI tools, editors, and resources used

### Frontmatter Template

```yaml
---
title: "Title in sentence case with quotes"
excerpt: "One-line description of the post"
tags: [Tag1, Tag2, Tag3]
comments: true
modified: YYYY-MM-DD
show_newsletter_signup: true
use_math: false
use_mermaid: false
toc: true
toc_label: Contents
toc_sticky: true
image: /images/image-name.jpg
header:
  image: /images/image-name.jpg
  caption: "Photo Credit: **Source**"
---
```

### Tone and Voice

The following analysis is drawn from close reading of all posts from 2025-2026. Use it to calibrate drafts.

#### Core stance

Michael writes as a practitioner sharing what he learned, not as an authority declaring truths. The implicit contract with the reader is: "I tried this, here's what happened, here's what I think it means." He is genuinely excited about AI but never uncritical. Every capability claim is grounded in a specific experience, and every limitation is stated plainly.

#### Sentence-level patterns

- **Short declarative sentences for emphasis, followed by longer explanatory ones.** "The bottleneck shifts. It's no longer 'how fast can I write code?' It's 'how many well-scoped tasks can I run simultaneously, and how quickly can I evaluate results?'"
- **Rhetorical questions used sparingly, to pivot.** "Could you run a Ralph loop to write a design doc? A grant proposal? A regulatory filing? Not yet."
- **Concrete before abstract.** Michael almost always opens a section with a specific incident or number, then extracts the principle. He earns abstractions; he does not lead with them.
- **Semicolons over em dashes.** Where most writers reach for an em dash to insert an aside, Michael uses a semicolon or splits into two sentences.
- **No throat-clearing.** Introductions open with action or an observation, never with meta-commentary about what the post will cover. "Last week I produced a 100+ page technical design doc in about three days." Not: "In this post, I want to explore..."
- **Natural burstiness.** Sentence length varies widely; 5-word punches alternate with 30-word explanatory sentences. This is genuine, not performed.

#### Paragraph-level patterns

- **One idea per paragraph, rigorously.** Paragraphs rarely exceed 4-5 sentences.
- **The pivot paragraph.** Michael often has a paragraph that begins with "But" or "And yet" that complicates the previous point. This is where his honesty about limitations lives. Example: "But the image is also misleading if you take it at face value."
- **Attribution is woven in, not dropped.** He names people and their ideas inline: "Ethan Mollick at Wharton recently made the same observation" rather than bare citations.
- **Lists used for concrete items (tools, steps), never for arguments.** Arguments are always in prose.

#### Section-level patterns

- **Introductions ground in personal experience.** Every post opens with a specific moment: sitting with his daughter, finishing a design doc, reflecting on a month at a new job. The personal anchor comes first; the thesis emerges from it.
- **Body sections follow a pattern: experience, then framework, then implication.** He describes what happened, names the pattern or framework it illustrates, then says what it means for the reader.
- **Conclusions are forward-looking, not summary.** They acknowledge uncertainty ("I don't know where this ends"), identify tensions, and point toward what comes next. They never mechanically restate section headings.

#### Recurring rhetorical moves

- **The "irony" turn.** Michael frequently identifies an irony or paradox in the situation. "And here's the irony: the skills that are so often dismissed as 'soft' turned out to be the hard ones." This is a signature move.
- **The honest caveat.** After describing something that worked well, he immediately names the thing that didn't. "Even though this remains my top complaint, there's a caveat here."
- **Quoting others to extend, not to defer.** He quotes Mollick, Karpathy, Cowen, etc. to build on their ideas, never as argument-from-authority. The quote opens a door; Michael walks through it with his own analysis.
- **The lived-experience proof.** Claims are backed by "I did X, and Y happened" rather than "studies show." When he does cite studies, they support a point he has already established through experience.

#### What Michael does NOT do

- No AI slop phrases: no "delve," "tapestry," "navigate the landscape," "in today's digital age"
- No excessive hedging: claims are stated directly, then bounded with specific qualifications
- No meta-commentary: no "it's worth noting," "it's important to mention," "let's discuss"
- No emoji in prose (occasional emoji in informal contexts like tweet embeds)
- No bullet-point arguments; bullets are reserved for tool lists, steps, and specifications
- No corporate jargon: no "leverage," "synergize," "drive innovation"
- No forced analogies or extended metaphors that aren't load-bearing
- Does not open posts with "time since I last posted" or similar weak hooks

#### Voice summary in one sentence

Michael writes like an engineer who reads economics and management theory, grounds everything in what he built last week, and trusts the reader to handle complexity without being talked down to.

### Code Snippets

- Use fenced code blocks with language specifiers (```python, ```yaml, etc.)
- Include relevant snippets that illustrate key concepts
- Keep snippets focused: show the important parts, not entire files
- Add brief explanations before or after code blocks

### Common Patterns

- **Transcript-style quotes**: For AI conversations, use code blocks with `@me:` and `@Gemini:` or similar
- **Embedded tweets**: Use Twitter/X embed code for social references
- **Internal links**: Reference other posts with relative URLs (`/blog/YYYY/MM/DD/slug/`)
- **Images**: Store in `/images/` with descriptive names

### Topics and Themes

Recent posts focus on:

- Agentic development workflows (Claude Code, Gemini, OpenCode)
- Context engineering and prompt design
- Practical AI tool usage and comparisons
- The "vibe coding" phenomenon with nuanced perspective
- Knowledge work and how AI is changing it

### Style

-  Avoid em dashes, as they can almost always be replaced by a semicolon
-  Prefer complete sentences
-  Keep sentences punchy
-  Avoid parenthetic asides or similar tangents within sentences; use two sentences instead
-  No bold taxonomy labels (e.g., "**Cost type:**", "**Benefit scaling:**") as section scaffolding; weave framework vocabulary into prose naturally
-  Avoid internal subsection headings (`###`) within short sections; continuous prose reads better unless the section is long enough to genuinely need navigation
-  Never fabricate anecdotes or lived experience; if the author hasn't provided a specific story, don't invent one
-  When referencing a person previously introduced by full title (e.g., "Our CTO Caleb Fenton"), subsequent references use first name only
-  Use "Delphos Labs" not "Delphos" when referring to the company
-  Attributed references over un-attributed statements; cite sources inline
-  Be precise about economic and technical terminology; don't use terms like "marginal analysis" unless the framework actually performs marginal analysis in the economic sense

## Project Structure

```
_posts/           # Blog posts in YYYY-MM-DD-slug.md format
images/           # Post images and assets
_layouts/         # Custom layouts (extends Minimal Mistakes)
assets/css/       # Custom CSS overrides
.github/
  scripts/        # Python scripts for automation
  workflows/      # GitHub Actions workflows
post-elements/    # Supporting docs (excluded from build)
```

## Newsletter System

The site has a MailerLite-powered newsletter with:

- **Sign-up form**: Embedded at the end of posts (controlled by `show_newsletter_signup: true`)
- **Automated draft creation**: GitHub Action creates MailerLite campaign drafts when new posts are pushed
- **Manual send**: Content is copied into MailerLite's block editor for final review

### Newsletter Workflow Files

- `.github/workflows/newsletter.yml` - Triggers on new posts in `_posts/`
- `.github/scripts/create_campaign.py` - Parses markdown and creates MailerLite draft

## Development

### Local Build

```bash
bundle exec jekyll serve
```

### Key Configuration

- `_config.yml` - Main Jekyll config
- `post-elements/` is excluded from the build
- `vendor/` is excluded from the build

## Working with This Repo

When drafting blog posts:

1. Follow the frontmatter template exactly
2. Start with `## Introduction` 
3. Use descriptive section headings
4. Include code snippets where relevant
5. End with `## Conclusion` and `## Tools Used`
6. Be honest about what worked and what didn't

When automating:

1. Prefer parsing markdown directly over building Jekyll
2. Use environment variables for secrets (MAILERLITE_API_KEY, etc.)
3. Keep Python scripts self-contained with minimal dependencies
4. Output clear, actionable information in workflow logs
