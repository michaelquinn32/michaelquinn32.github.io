#!/usr/bin/env python3
"""Create a MailerLite draft campaign for a blog post."""

import json
import os
import sys
import urllib.request
import urllib.error


def create_campaign(
    api_key: str,
    title: str,
    from_name: str,
    from_email: str,
    reply_to: str,
    html_content: str,
) -> str | None:
    """Create a draft campaign in MailerLite with content.
    
    Arguments:
        api_key: MailerLite API key.
        title: Campaign/email subject.
        from_name: Sender name.
        from_email: Sender email address.
        reply_to: Reply-to email address.
        html_content: Full HTML content for the email.
    
    Returns:
        Campaign ID if successful, None otherwise.
    """
    url = "https://connect.mailerlite.com/api/campaigns"
    
    data = {
        "name": f"Blog Post: {title}",
        "type": "regular",
        "emails": [
            {
                "subject": title,
                "from_name": from_name,
                "from": from_email,
                "reply_to": reply_to,
                "content": html_content,
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


def build_email_html(title: str, post_url: str, article_content: str) -> str:
    """Build email-friendly HTML wrapper for article content.
    
    Arguments:
        title: Post title.
        post_url: Full URL to the post.
        article_content: Extracted article HTML.
    
    Returns:
        Complete HTML document for the email.
    """
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
    h1, h2, h3 {{ color: #1a1a1a; }}
    a {{ color: #0066cc; }}
    pre {{ background: #f5f5f5; padding: 15px; overflow-x: auto; border-radius: 4px; }}
    code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
    pre code {{ background: none; padding: 0; }}
    img {{ max-width: 100%; height: auto; }}
    blockquote {{ border-left: 4px solid #ddd; margin-left: 0; padding-left: 20px; color: #666; }}
  </style>
</head>
<body>
  <p style="color: #666; font-size: 0.9em;">New post from Michael Quinn</p>
  <h1><a href="{post_url}">{title}</a></h1>
  
  {article_content}
  
  <hr style="margin: 40px 0; border: none; border-top: 1px solid #ddd;">
  <p style="color: #666; font-size: 0.9em;">
    <a href="{post_url}">Read on the web</a> | 
    <a href="{{$unsubscribe}}">Unsubscribe</a>
  </p>
</body>
</html>"""


def main():
    """Main entry point."""
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <title> <post_url> <article_html_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    title = sys.argv[1]
    post_url = sys.argv[2]
    article_html_file = sys.argv[3]
    
    # Get config from environment
    api_key = os.environ.get("MAILERLITE_API_KEY")
    if not api_key:
        print("Error: MAILERLITE_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    from_name = os.environ.get("FROM_NAME", "Michael Quinn")
    from_email = os.environ.get("FROM_EMAIL", "hello@msquinn.com")
    reply_to = os.environ.get("REPLY_TO", "msquinn.illinois@gmail.com")
    
    # Read article content
    with open(article_html_file, "r", encoding="utf-8") as f:
        article_content = f.read()
    
    # Build full email HTML
    email_html = build_email_html(title, post_url, article_content)
    
    # Create campaign with content
    print(f"Creating campaign: {title}")
    campaign_id = create_campaign(
        api_key, title, from_name, from_email, reply_to, email_html
    )
    
    if not campaign_id:
        print("Failed to create campaign", file=sys.stderr)
        sys.exit(1)
    
    print(f"Created campaign with ID: {campaign_id}")
    print(f"Campaign URL: https://dashboard.mailerlite.com/campaigns/{campaign_id}")
    print("Campaign is ready for review in MailerLite dashboard.")


if __name__ == "__main__":
    main()
