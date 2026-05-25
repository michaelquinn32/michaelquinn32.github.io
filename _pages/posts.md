---
title: Recent Posts
excerpt: "Writing on engineering leadership, AI-native development, and the quantitative thinking underneath."
permalink: /blog/
layout: single
author_profile: true
header:
  image: /images/main_header.jpg
  caption: "Photo credit: Dan Quinn"
classes: wide
---

{% for post in site.posts %}
{% assign year = post.date | date: "%Y" | plus: 0 %}
{% if year >= 2020 %}

<div class="list__item">
  <article class="archive__item">
    <h2 class="archive__item-title no_toc" itemprop="headline">
      <a href="{{ post.url | relative_url }}" rel="permalink">{{ post.title }}</a>
    </h2>
    <p class="page__meta">{{ post.date | date: "%B %d, %Y" }}</p>
    {% if post.excerpt %}<p class="archive__item-excerpt" itemprop="description">{{ post.excerpt | markdownify | strip_html | truncate: 200 }}</p>{% endif %}
  </article>
</div>
  {% endif %}
{% endfor %}

---

## Earlier writing (2014-2016)

R, statistics, and functional programming from a previous life.

{% for post in site.posts %}
{% assign year = post.date | date: "%Y" | plus: 0 %}
{% if year < 2020 %}

- [{{ post.title }}]({{ post.url | relative_url }}) — {{ post.date | date: "%B %Y" }}
  {% endif %}
  {% endfor %}
