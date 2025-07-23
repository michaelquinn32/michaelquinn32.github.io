This guide will walk you through modifying your Jekyll site (using the Minimal Mistakes theme) to show a MailerLite newsletter signup form at the end of posts based on a front matter variable.

Step 1: Add the MailerLite Universal Script
First, you need to add the main MailerLite JavaScript to every page on your site. The Minimal Mistakes theme makes this easy and clean by providing a dedicated file for custom head scripts.

In the root directory of your Jekyll site, navigate to the _includes folder. If it doesn't exist, create it.

Inside _includes, create a head folder.

Inside the _includes/head/ folder, create a new file named custom.html.

Paste the MailerLite Universal script directly into this custom.html file.

Your file at _includes/head/custom.html should contain only this:

```
<!-- MailerLite Universal -->
<script>
    (function(w,d,e,u,f,l,n){w[f]=w[f]||function(){(w[f].q=w[f].q||[])
    .push(arguments);},l=d.createElement(e),l.async=1,l.src=u,
    n=d.getElementsByTagName(e)[0],n.parentNode.insertBefore(l,n);})
    (window,document,'script','https://assets.mailerlite.com/js/universal.js','ml');
    ml('account', '1679653');
</script>
<!-- End MailerLite Universal -->
```

The theme will automatically include this script before the closing </head> tag on every page. You only need to do this step once.

Step 2: Modify the Post Layout to Show the Form
Now, you'll modify the layout file that renders your blog posts to check for the show_newsletter_signup variable.

In the root of your site, create a folder named _layouts if it doesn't already exist.

Create a new file inside this folder named single.html.

Copy and paste the code below into your new _layouts/single.html file. This code is the standard layout for a post in Minimal Mistakes, with our conditional logic added near the end of the content section.

```
---
layout: default
---

{%- assign_pages = site.pages | where: "type", "article" -%}
{%- assign_posts = site.posts | where: "type", "article" -%}
{%- assign_articles = assign_pages | concat: assign_posts -%}

{% if page.id %}
  {% assign post = articles | where: "id", page.id | first %}
{% else %}
  {% assign post = articles | where: "slug", page.slug | first %}
{% endif %}

<div id="main" role="main" class="container">
  <article class="page h-entry" itemscope itemtype="https://schema.org/CreativeWork">
    {% if page.title %}<meta itemprop="headline" content="{{ page.title | markdownify | strip_html | strip_newlines | escape_once }}">{% endif %}
    {% if page.excerpt %}<meta itemprop="description" content="{{ page.excerpt | markdownify | strip_html | strip_newlines | escape_once }}">{% endif %}
    {% if page.date %}<meta itemprop="datePublished" content="{{ page.date | date_to_xmlschema }}">{% endif %}
    {% if page.last_modified_at %}<meta itemprop="dateModified" content="{{ page.last_modified_at | date_to_xmlschema }}">{% endif %}

    <div class="page__inner-wrap">
      {% unless page.header.overlay_color or page.header.overlay_image %}
        <header>
          {% if page.title %}<h1 id="page-title" class="page__title p-name" itemprop="headline">{{ page.title | markdownify | remove: "<p>" | remove: "</p>" }}</h1>{% endif %}
          {% include page__meta.html %}
        </header>
      {% endunless %}

      <section class="page__content e-content" itemprop="text">
        {{ content }}

        <!-- --- BEGIN CONDITIONAL NEWSLETTER FORM --- -->
        {% if page.show_newsletter_signup %}
          <hr>
          <div class="text-center">
            <h3>Subscribe to the Newsletter</h3>
            <p>Get posts like this delivered right to your inbox.</p>
          </div>
          <div class="ml-embedded" data-form="c4vDL8"></div>
        {% endif %}
        <!-- --- END CONDITIONAL NEWSLETTER FORM --- -->

        {% if page.link %}<div><a href="{{ page.link }}" class="btn btn--primary">{{ site.data.ui-text[site.locale].ext_link_label | default: "Direct Link" }}</a></div>{% endif %}
      </section>

      <footer class="page__meta">
        {% if site.data.ui-text[site.locale].meta_label %}
          <h4 class="page__meta-title">{{ site.data.ui-text[site.locale].meta_label }}</h4>
        {% endif %}
        {% include page__taxonomy.html %}
        {% include page__date.html %}
      </footer>

      {% if page.share %}{% include social-share.html %}{% endif %}

      {% include post_pagination.html %}
    </div>

    {% if jekyll.environment == 'production' and site.comments.provider and page.comments %}
      {% include comments.html %}
    {% endif %}
  </article>

  {% comment %}<!-- only show related on a post page when `related: true` -->{% endcomment %}
  {% if page.id and page.related and site.related_posts.size > 0 %}
    <div class="page__related">
      <h2 class="page__related-title">{{ site.data.ui-text[site.locale].related_label | default: "You May Also Enjoy" }}</h2>
      <div class="grid__wrapper">
        {% for post in site.related_posts limit:4 %}
          {% include archive-single.html type="grid" %}
        {% endfor %}
      </div>
    </div>
  {% endif %}
</div>
```

The key section is the one marked BEGIN CONDITIONAL NEWSLETTER FORM. It checks if page.show_newsletter_signup is true. If it is, it prints a horizontal rule, a heading, and then the MailerLite form div.

Step 3: Enable the Form in Your Blog Posts
Now for the easy part! In any post where you want the form to appear, simply add show_newsletter_signup: true to the front matter.

Here is an example of a post's Markdown file:

```
---
title: "My First Post with a Newsletter"
date: 2025-07-20
tags: [Jekyll, Tutorial]
show_newsletter_signup: true
---

Welcome to my blog post. Here is some interesting content that will make people want to subscribe to my newsletter.

### A Sub-heading

More fascinating content goes here. When the reader gets to the end, they will see the signup form because of the front matter variable we set.

For any posts where you omit the show_newsletter_signup: true line or set it to false, the form will not be displayed.

This setup gives you precise, per-post control over your newsletter form, all managed through the simple front matter you're already using.
```
