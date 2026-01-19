#!/usr/bin/env python3
"""Create a MailerLite draft campaign for a blog post.

Parses frontmatter from markdown and outputs fields for manual entry
into MailerLite's block editor (free tier doesn't support API HTML content).
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def parse_frontmatter(markdown_path: str) -> tuple[dict, str]:
    """Parse YAML frontmatter and content from a markdown file.

    Arguments:
        markdown_path: Path to the markdown file.

    Returns:
        Tuple of (frontmatter dict, content string after frontmatter).
    """
    with open(markdown_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on --- delimiters
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    frontmatter_text = parts[1].strip()
    body = parts[2].strip()

    # Simple YAML parsing (handles our specific frontmatter format)
    frontmatter = {}
    current_key = None
    current_value = None

    for line in frontmatter_text.split("\n"):
        # Check for nested key (indented)
        if line.startswith("  ") and current_key:
            nested_match = re.match(r"^\s+(\w+):\s*(.*)$", line)
            if nested_match:
                nested_key = nested_match.group(1)
                nested_val = nested_match.group(2).strip().strip('"').strip("'")
                if isinstance(frontmatter.get(current_key), dict):
                    frontmatter[current_key][nested_key] = nested_val
            continue

        # Check for top-level key
        match = re.match(r"^(\w+):\s*(.*)$", line)
        if match:
            current_key = match.group(1)
            value = match.group(2).strip()

            # Handle arrays like [AI, Agents]
            if value.startswith("[") and value.endswith("]"):
                current_value = [
                    v.strip().strip('"').strip("'")
                    for v in value[1:-1].split(",")
                ]
            elif value == "":
                # Might be a nested dict
                current_value = {}
            else:
                current_value = value.strip('"').strip("'")

            frontmatter[current_key] = current_value

    return frontmatter, body


def extract_intro_paragraphs(markdown_content: str, max_paragraphs: int = 3) -> str:
    """Extract intro paragraphs before first ## heading.

    Arguments:
        markdown_content: Markdown content after frontmatter.
        max_paragraphs: Maximum number of paragraphs to extract.

    Returns:
        Intro paragraphs as plain text.
    """
    # Find content before first ## heading (but after any initial ## Introduction)
    lines = markdown_content.split("\n")

    intro_lines = []
    found_intro_heading = False
    in_code_block = False

    for line in lines:
        # Track code blocks
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        # Skip the ## Introduction heading itself
        if line.strip() == "## Introduction":
            found_intro_heading = True
            continue

        # Stop at next ## heading
        if line.startswith("## ") and found_intro_heading:
            break

        # Skip image lines
        if line.startswith("!["):
            continue

        intro_lines.append(line)

    # Join and split into paragraphs
    text = "\n".join(intro_lines).strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Take up to max_paragraphs
    selected = paragraphs[:max_paragraphs]

    # Clean up markdown formatting for plain text
    result = "\n\n".join(selected)

    # Remove markdown links but keep text: [text](url) -> text
    result = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", result)

    # Remove emphasis markers
    result = result.replace("**", "").replace("*", "").replace("_", "")

    return result


def calculate_reading_time(markdown_content: str, wpm: int = 200) -> int:
    """Calculate reading time in minutes.

    Arguments:
        markdown_content: Full markdown content.
        wpm: Words per minute reading speed.

    Returns:
        Reading time in minutes (minimum 1).
    """
    # Remove code blocks
    content = re.sub(r"```.*?```", "", markdown_content, flags=re.DOTALL)

    # Remove images
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

    # Count words
    words = len(content.split())

    return max(1, round(words / wpm))


def format_date_from_filename(filename: str) -> str:
    """Extract and format date from post filename.

    Arguments:
        filename: Post filename like 2025-08-31-slug.md

    Returns:
        Formatted date like "August 31, 2025"
    """
    # Extract YYYY-MM-DD from filename
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if not match:
        return ""

    year, month, day = match.groups()
    date = datetime(int(year), int(month), int(day))
    return date.strftime("%B %d, %Y")


def build_post_url(filename: str, site_url: str) -> str:
    """Build post URL from filename.

    Arguments:
        filename: Post filename like 2025-08-31-slug.md
        site_url: Base site URL.

    Returns:
        Full post URL.
    """
    # Extract components from filename
    name = Path(filename).stem
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})-(.+)", name)
    if not match:
        return ""

    year, month, day, slug = match.groups()
    return f"{site_url}/blog/{year}/{month}/{day}/{slug}/"


def get_image_url(frontmatter: dict, site_url: str) -> str:
    """Extract hero image URL from frontmatter.

    Arguments:
        frontmatter: Parsed frontmatter dict.
        site_url: Base site URL.

    Returns:
        Full image URL or empty string.
    """
    # Try header.image first, then image
    image_path = ""
    if isinstance(frontmatter.get("header"), dict):
        image_path = frontmatter["header"].get("image", "")
    if not image_path:
        image_path = frontmatter.get("image", "")

    if not image_path:
        return ""

    # Make absolute URL
    if image_path.startswith("/"):
        return f"{site_url}{image_path}"
    elif image_path.startswith("http"):
        return image_path
    else:
        return f"{site_url}/{image_path}"


def create_campaign(
    api_key: str,
    subject: str,
    from_name: str,
    from_email: str,
    reply_to: str,
) -> str | None:
    """Create a draft campaign in MailerLite (without content).

    Arguments:
        api_key: MailerLite API key.
        subject: Email subject line.
        from_name: Sender name.
        from_email: Sender email address.
        reply_to: Reply-to email address.

    Returns:
        Campaign ID if successful, None otherwise.
    """
    url = "https://connect.mailerlite.com/api/campaigns"

    data = {
        "name": subject,
        "type": "regular",
        "emails": [
            {
                "subject": subject,
                "from_name": from_name,
                "from": from_email,
                "reply_to": reply_to,
            }
        ],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            campaign_id = result.get("data", {}).get("id")
            return campaign_id
    except urllib.error.HTTPError as e:
        print(f"Error creating campaign: {e}", file=sys.stderr)
        print(e.read().decode("utf-8"), file=sys.stderr)
        return None


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <markdown_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    markdown_path = sys.argv[1]

    # Get config from environment
    api_key = os.environ.get("MAILERLITE_API_KEY")
    if not api_key:
        print(
            "Error: MAILERLITE_API_KEY environment variable not set",
            file=sys.stderr,
        )
        sys.exit(1)

    site_url = os.environ.get("SITE_URL", "https://www.msquinn.com")
    from_name = os.environ.get("FROM_NAME", "Michael Quinn")
    from_email = os.environ.get("FROM_EMAIL", "hello@msquinn.com")
    reply_to = os.environ.get("REPLY_TO", "msquinn.illinois@gmail.com")

    # Parse the markdown file
    frontmatter, content = parse_frontmatter(markdown_path)

    # Extract all the fields
    title = frontmatter.get("title", "New Post")
    excerpt = frontmatter.get("excerpt", "")
    filename = Path(markdown_path).name
    post_url = build_post_url(filename, site_url)
    image_url = get_image_url(frontmatter, site_url)
    formatted_date = format_date_from_filename(filename)
    reading_time = calculate_reading_time(content)
    intro_paragraphs = extract_intro_paragraphs(content)

    # Create the campaign
    print(f"Creating campaign: {title}")
    campaign_id = create_campaign(
        api_key, title, from_name, from_email, reply_to
    )

    if not campaign_id:
        print("Failed to create campaign", file=sys.stderr)
        sys.exit(1)

    campaign_url = (
        f"https://dashboard.mailerlite.com/campaigns/{campaign_id}/content"
    )

    # Output the fields in a clean, copy-paste friendly format
    print("")
    print("=" * 70)
    print("CAMPAIGN CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Campaign URL: {campaign_url}")
    print("")
    print("Copy the fields below into MailerLite's block editor:")
    print("")
    print("-" * 70)
    print("POST URL (for button/link):")
    print("-" * 70)
    print(post_url)
    print("")
    print("-" * 70)
    print("HERO IMAGE URL:")
    print("-" * 70)
    print(image_url if image_url else "(no image in frontmatter)")
    print("")
    print("-" * 70)
    print("HEADLINE / TITLE:")
    print("-" * 70)
    print(title)
    print("")
    print("-" * 70)
    print("EXCERPT (for preview text / subtitle):")
    print("-" * 70)
    print(excerpt if excerpt else "(no excerpt in frontmatter)")
    print("")
    print("-" * 70)
    print(f"DATE & READING TIME:")
    print("-" * 70)
    print(f"{formatted_date} · {reading_time} min read")
    print("")
    print("-" * 70)
    print("INTRO PARAGRAPHS (main content before 'Read more'):")
    print("-" * 70)
    print(intro_paragraphs)
    print("")
    print("=" * 70)
    print("NEXT STEPS:")
    print("1. Click the Campaign URL above")
    print("2. Design the email using the block editor")
    print("3. Copy/paste the fields above into the appropriate blocks")
    print("4. Preview and send when ready")
    print("=" * 70)


if __name__ == "__main__":
    main()
