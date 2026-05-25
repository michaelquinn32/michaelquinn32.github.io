---
title: "I lead engineering teams and ship alongside them."
layout: splash
date: 2016-03-23T11:48:41-04:00
header:
  overlay_image: /images/tree-desert.jpg
  overlay_color: "#000"
  overlay_filter: "0.25"
excerpt: "Player coach. Engineering leader. Ex-Google."
classes:
  - landing
  - dark-theme
feature_row_about:
  - image_path: /images/desert-portrait.jpg
    alt: "Hi! I'm Michael"
    title: "Why me?"
    excerpt: >-
      I lead the Security Research, Infrastructure Full-stack and Data Science teams at 
      [Delphos Labs](https://delphoslabs.com/). We're automating reverse engineering and
      solving the next generation of cybersecurity problems. Before Delphos Labs, I spent
      eight years at Google working on AI; most recently on data acquisition and quality 
      for flagship AI products.
    url: "/about-me/"
    btn_label: "Read More"
    btn_class: "btn--primary"
feature_row_engagements:
  - image_path: /images/services/glacier.jpg
    alt: "fractional cto"
    title: "Fractional CTO"
    excerpt: >-
      10-20 hours per week. Architecture decisions, hiring, technical strategy,
      code review, and writing code where it matters. For seed-stage teams who
      need engineering leadership before they can hire it full-time.
  - image_path: /images/services/canyon.jpg
    alt: "technical advisor"
    title: "Technical advisor"
    excerpt: >-
      Lower-touch, advisory shares. Architecture sounding board, hiring help,
      AI strategy, and periodic deep-dives on specific decisions.
  - image_path: /images/services/mountain.jpg
    alt: "zero to one audit"
    title: "0-to-1 audits"
    excerpt: >-
      One-week paid sprints. Architecture review, AI capability assessment,
      or a concrete recommendation on what to build next.
---

{% include feature_row id="feature_row_about" type="left" %}

Recent posts on [agent orchestration](/blog/2026/02/14/agents-all-the-way-down/), [the new shape of knowledge work](/blog/2026/01/17/knowledge-stack/), and [the economics of abundant software](/blog/2026/04/18/marginal-analysis/).
{: .text-center}

---

<h1 class="archive__item-title">Engagements</h1>

{% include feature_row id="feature_row_engagements" %}

<div class="feature__wrapper contact-section">
  <div class="feature__item--left">
    <div class="archive__item">
      <div class="archive__item-teaser">
        <img src="/images/path.jpg" alt="Let's work together.">
      </div>
      <div class="archive__item-body">
        <form
          action="https://formspree.io/f/xwpbkkyb"
          class="fs-form"
          target="_top"
          method="POST"
        >
          <h2 class="archive__item-title" style="padding-bottom: 0; margin: 0;">Let's talk</h2>
          <div class="archive__item-excerpt">
            <p>Tell me what you're working on and what's hard about it.</p>
          </div>
          <div class="fs-field">
            <label class="fs-label" for="email">Your Email</label>
            <input class="fs-input" id="email" name="email" required />
          </div>
          <div class="fs-field">
            <label class="fs-label" for="message">What are you building?</label>
            <textarea
              class="fs-textarea"
              id="message"
              name="message"
              required
            ></textarea>
          </div>
          <div class="fs-button-group">
            <button class="fs-button" type="submit">Send</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</div>
