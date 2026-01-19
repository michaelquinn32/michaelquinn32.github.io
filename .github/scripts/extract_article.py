#!/usr/bin/env python3
"""Extract article content from rendered Jekyll HTML for email newsletter."""

import re
import sys


def extract_article(html_path: str) -> str:
    """Extract the main article content from a rendered Jekyll page.
    
    Arguments:
        html_path: Path to the rendered HTML file.
    
    Returns:
        Cleaned HTML content suitable for email.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try to extract just the article content
    # Look for <article> tag first
    article_match = re.search(r"<article[^>]*>(.*?)</article>", content, re.DOTALL)
    if article_match:
        article_html = article_match.group(1)
    else:
        # Fallback: look for main content area (minimal-mistakes theme)
        main_match = re.search(
            r'<div class="page__inner-wrap">(.*?)</div>\s*</div>\s*</article>',
            content,
            re.DOTALL,
        )
        if main_match:
            article_html = main_match.group(1)
        else:
            # Last resort: use body content
            body_match = re.search(r"<body[^>]*>(.*?)</body>", content, re.DOTALL)
            article_html = body_match.group(1) if body_match else content

    # Clean up: remove navigation, scripts, etc.
    article_html = re.sub(r"<nav[^>]*>.*?</nav>", "", article_html, flags=re.DOTALL)
    article_html = re.sub(
        r"<script[^>]*>.*?</script>", "", article_html, flags=re.DOTALL
    )
    article_html = re.sub(r"<aside[^>]*>.*?</aside>", "", article_html, flags=re.DOTALL)

    return article_html.strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <html_file>", file=sys.stderr)
        sys.exit(1)

    html_path = sys.argv[1]
    print(extract_article(html_path))
