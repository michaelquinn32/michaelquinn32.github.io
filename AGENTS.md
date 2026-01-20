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

- **First person narrative**: "I wanted to...", "My approach was..."
- **Conversational but technical**: Explains concepts without dumbing them down
- **Honest about limitations**: Admits when things don't work or require workarounds
- **Practical focus**: Emphasizes real workflows and tangible outcomes
- **No excessive hype**: Balanced view of AI capabilities and limitations

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
