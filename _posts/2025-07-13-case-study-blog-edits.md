---
title: "Case study: Creating a newsletter for this blog using Gemini"
excerpt: "Documenting a somewhat realistic workflow as I update my blog."
tags: [Claude, VS Code, MCP, AI, Development]
comments: true
modified: 2025-07-23
show_newsletter_signup: true
use_math: true
use_mermaid: false
toc: true
toc_label: Contents
toc_sticky: true
image: /images/2025-07-13-case-study/hiker.png
header:
  image: /images/2025-07-13-case-study/hiker.png
  caption: "Photo Credit: [**Playground.com**](https://playground.com/design/surreal-hiker-journeying-through-whimsical-dreamlike-landscape-cmd2ly2mc00003j6z4zsp56or)"
---

## Introduction

This site is hosted directly from my GitHub repository using [GitHub Pages](https://pages.github.com/). I've written before about how this is all built:

- The [original version](/blog/2015/11/30/how-i-built-this-site/)
- And the more [modern version](/blog/2025/03/06/building-with-devcontainers/)

I honestly don't know where all this is going, but it would be nice to support a newsletter for this blog. Since this involves a bit of coding, I thought it would be a good opportunity to document the process and narrate the experience of using an agent to get this done.

I hope to accomplish a few things:

- Get a newsletter (yay!)
- Interact with a new agent I haven't used previously: the new [Gemini Code Assist](https://codeassist.google/) (not to be confused with the Gemini App)
- Narrate how this all goes, giving you a taste of agentic development

One particularly exciting aspect: most of these things I've never implemented before. I am definitely not a web developer. Being able to rely on Gemini's unique skills is very neat, and this will be a nice illustration of how an agent can raise your skill level, especially in a new domain.

## The Plan (for a Plan)

Here's the plan:

- Use Gemini Code Assist and Gemini CLI to do most of the work (they need to be installed)
- Establish a plan for transforming this into a newsletter
- Using Gemini App, get a detailed path to implementation (assuming this will take a few turns)
- Start iterating on that plan with Gemini Code Assist

These planned steps follow a rough version of the standard software development loop:

- Frame the problem
- Design the solution
- Write code
- Launch

Real work would require a few more steps, especially tests. I'm not going to write any actual tests for this—sorry, it's a side project. Instead, I have a way to build and serve the site [locally](/blog/2025/03/06/building-with-devcontainers/), and that will ultimately be my test for whether this actually works.

## Aside: Agentic Work as Context Engineering

My approach to solving this problem serves an additional purpose: to illustrate [context engineering](https://blog.langchain.com/context-engineering-for-agents/) and how it plays a key role in agentic development workflows. Context engineering is the evolution of prompt engineering, which we could colloquially describe as "creating plans for and providing background to agents." The agent can accomplish the task as long as you enough relevant information and can structure your task in a way that it can be executed. In many surprising ways, it's not all that different from working with other people—if those people needed to be onboarded to every new task they were given.

[Phil Schmid](https://www.philschmid.de/context-engineering) of DeepMind has a great description of practical context engineering. I recommend checking out the whole thing. In the meantime, I want to highlight this diagram he produced:

![what-is-context](/images/2025-07-13-case-study/context.png)

As Phil explains, _context_ consists of all of these different things:

- Knowledge baked into the model
- Directions for a particular task
- A model "state" or more detailed tasks descriptions
- Information from tools and other sources

Gemini gives us several opportunities to build context for this project:

- The aforementioned brainstorm within Gemini App
- This blog post, which Gemini can "read" when I interact with Code Assist
- Gemini's ability to search online and fetch web content
- Gemini's ability to use the repo of this site as context, helpfully accessed with the `@` character in chat

![useful-stuff-in-the-menu](/images/2025-07-13-case-study/gemini-tools-menu.png)

I'll be sure to highlight these different context-building approaches as we go. Let's get started!

## Setting Up: Gemini Code Assist and Gemini CLI

Work pays for Claude Code, so I might as well try something different for this site. As a Google Pixel owner, I already have a Gemini Pro [subscription](https://gemini.google/subscriptions/), and I've been pretty happy with 2.5 Pro so far. Unfortunately, code tools aren't included in that plan, but this is a small project, so it's not a big deal.

Installing requires:

- [Disabling](https://github.com/orgs/community/discussions/147437) Copilot in VS Code
- Installing the [Gemini Code Assist](https://marketplace.visualstudio.com/items?itemName=Google.geminicodeassist) extension
- Installing the [Gemini CLI](https://github.com/google-gemini/gemini-cli)

## Context Building: Drafting a Design to Solve the Problem

Full disclosure: I don't really know how to solve this problem. I'll start from the fact that my [main page](https://michaelquinn32.github.io/) has an intake form. Hopefully, the service powering this form ([formspree.io](https://formspree.io)) can be used to also collect email addresses, and I'll leverage the fact that my site has an RSS feed to put together some system for content distribution. The fact that these ideas are a little fuzzy is no big deal—agents are here to help.

The following edited transcript comes from a conversation I had in the [Gemini App](https://g.co/gemini/share/e7ad71bfa204), where I explored options for developing a newsletter using my existing static site hosted on GitHub Pages.

```
@me: I want to use tools that currently run my website
to deliver a newsletter. *Lots of details on the site*.
How do I do this?

@Gemini: That can work, but you'll need to use a service
like MailerLite or Mailchimp.

@me: Mailchimp RSS-to-email campaigns seem like a premium
feature. Provide a breakdown of different free options.

@Gemini: Here's a table of options, leaning heavily into
MailerLite as a solution.

... A couple of iterations, including some recommendations
from me after web searches ...

@Gemini: I really want you to use MailerLite.
```

Okay, fine. I signed up for the MailerLite free plan and used the editor on their website to create a form.

```
@me: I've created a form and copied the code below. How do I
modify my website to show this form at the end of blog posts?
I want to use markdown front matter to control if the form appears.

@Gemini: Sure! Here's a Canvas that contains everything you'll
need to do.
```

I took the [Canvas](https://gemini.google/overview/canvas/?hl=en) it produced and copied it into a separate markdown file, which I'll let the agent follow to make the necessary changes for this site. You can see that file on my [GitHub](https://github.com/michaelquinn32/michaelquinn32.github.io/tree/master/post-elements/NEWSLETTER_SETUP.md).

## Getting to Work

Time to set the agent to work. I created two basic sources of context for the project:

- This blog post, which I was feeding into the model while drafting
- The file I developed in the previous section, called `NEWSLETTER_SETUP.md`

Opening Code Assist in VS Code, I entered the following prompt. Remember `@file` tells Gemini to read that particular file:

```
@2025-07-13-case-study-blog-edits.md
@NEWSLETTER_SETUP.md

Use the current content of the blog post and the instructions
in the markdown file to add a sign-up form that can be controlled
using front matter.
```

With that prompt, Gemini proposed a series of changes. One was to create a new layout file for posts that would support my desire to use front matter to show the sign-up form. The other was to change the current post as I was drafting it.

![changes-proposed-by-gemini](/images/2025-07-13-case-study/gemini-changes.png)

Clicking the checkmark accepted the changes and edited my files. It was nice to view the diffs Gemini was proposing and confirm these were the changes I wanted.

## Debugging, Iterating, and Improving

As I mentioned, my test is whether the site builds and looks correct. As my post on developing the site [mentioned](/blog/2025/03/06/building-with-devcontainers/), I can build and serve the site with `bundle exec jekyll serve`. The first attempt to build the site with the new files didn't work, and it was up to Gemini to fix the issue.

> **Note**: When working with Claude I would have the agent call my particular tools as well, but I decided to go the manual route for this project.

```
$ bundle exec jekyll serve
To use retry middleware with Faraday v2.0+, install `faraday-retry` gem
/usr/local/rvm/gems/ruby-3.4.4/gems/jekyll-algolia-1.7.1/lib/jekyll/algolia/progress_bar.rb:4: warning: ostruct was loaded from the standard library, but will no longer be part of the default gems starting from Ruby 3.5.0.
You can add ostruct to your Gemfile or gemspec to silence this warning.
Configuration file: /workspaces/michaelquinn32.github.io/_config.yml
            Source: /workspaces/michaelquinn32.github.io
       Destination: /workspaces/michaelquinn32.github.io/_site
 Incremental build: disabled. Enable with --incremental
      Generating...
      Remote Theme: Using theme mmistakes/minimal-mistakes
       Jekyll Feed: Generating feed for posts
  Liquid Exception: Liquid syntax error (line 1): Unknown tag 'assign_pages' in /_layouts/single.html
```

I pasted this into Code Assist to see what Gemini thought. It proposed some edits to `/_layouts/single.html`, but this revealed another error. My plan to build the newsletter contained liquid tags, and my build script was somehow picking that up. Another round of debugging.

```
The NEWSLETTER_SETUP.md file shouldn't be part of the site,
but it appears that Jekyll is trying to render it. I'm getting
an error about liquid tags in that file.

How do I exclude it from the site?
```

This time, Gemini proposed changes to my `_config.yaml` file. I accepted these changes and tried again. The site built! But all the content was missing! I needed to do more debugging, and [Chrome DevTools](https://developer.chrome.com/docs/devtools) revealed that I hadn't closed an HTML comment. Oops. I applied that fix manually.

After more visual inspection of the locally running site, there seemed to be other issues. Elements that were supposed to be on all pages, like my author bio, weren't rendering. This seemed tied to Gemini's interpretation of Minimal Mistakes' original [layout](https://github.com/mmistakes/minimal-mistakes/blob/master/_layouts/single.html). I grabbed another copy from GitHub, pasted it into the prompt, and asked for another fix.

```
When I look at the site after local rendering, my author bio
and some other post elements are missing. Fetch the current
layout from Minimal Mistakes here:
https://github.com/mmistakes/minimal-mistakes/blob/master/_layouts/single.html

Fix the site and keep the change to include the form.
```

Here, I want to highlight that Gemini has a web content [fetch tool](https://developers.google.com/gemini-code-assist/docs/gemini-cli#gemini-code-assist-agent-mode). You might remember that I discussed tools in a previous post on [MCPs](/blog/2025/06/22/mcps/)

This actually took two attempts because I exceeded the output limit from Code Assist. Fixing this required removing some no-longer-needed files from context. Nonetheless, it was able to update the `single.html` layout. I accepted its suggested edits.

## Some Bigger Challenges: Making the Form Look Nice

Moving on to harder problems. The original version of the form didn't look great and had a strange visual artifact floating in the background.

![weird-form](/images/2025-07-13-case-study/weird-form.jpeg)

_What in the world is that gross little black box doing there?_

To fix this, I needed to create more prompts. This time, I told Gemini to refer to the full HTML form provided by [MailerLite](https://www.mailerlite.com/features/signup-forms) and update the CSS to match what I wanted. I also included the screenshot of the weird visual artifact from my site. To include a screenshot, I used `Ctrl + Shift + S` in Edge and copied the saved file into my working directory.

One more big prompt:

```
@single.html @Screenshot_20-7-2025_19950_127.0.0.1.jpeg
@form-html @NEWSLETTER_SETUP.md

There's a strange visual artifact on the form when I use the
MailerLite JavaScript approach. I've included the full HTML
option.

- In single.html, remove
  `<div class="ml-embedded" data-form="c4vDL8"></div>`
  and add enough HTML to render the form
- Separate the CSS into its own file in @assets
- Fix the form so it doesn't have a visual artifact
- Increase the width of the form so it matches the article width
```

Code Assist failed at this point. I ran into the `code sample in this response was truncated` [error](https://www.reddit.com/r/GeminiAI/comments/1lq4n8y/gemini_code_assist_in_vs_code_weird_output_limits/) multiple times. To get around this, I tried the Gemini CLI instead. I used the same prompt. The CLI tool spent a very long time working on the problem—more than five minutes for the first prompt that only processed the context. It told jokes as it went, like "Why don't programmers like nature? Too many bugs." I'm really unsure whether this is a good user experience.

That said, on the second iteration of the prompt, Gemini CLI actually got to work. There was still debugging to do, building the site and copying in errors. But it was able to fix my issue after a couple of iterations. I still needed to inspect the built page and use DevTools to find bad elements and paste those into the prompt. The last fix I made looked like this:

```
Looking at the site again, my form.ml-block-form
is still dark. Here's the beginning of the HTML for this.
Fix the styling for this.

<form class="ml-block-form" action="..." data-code=""
method="post" target="_blank">
```

After failing a couple of times, I was able to use Gemini to give me the pointers I needed to edit the CSS myself. 🙃 Still, I got what I wanted, which is the nice form you see at the bottom.

## Conclusion

While I ultimately reached my desired destination, there were fits and starts along the way:

- Gemini Code Assist couldn't handle the amount of context I wanted to include
- Gemini CLI got almost to the end
- But I had to use the edits Gemini was attempting to make the manual changes I wanted

From the perspective of a blog post, this is a great outcome because this is exactly what professional context engineering and agentic development looks like. In fact, it seems a lot like working with an _eager, forgetful, and sometimes mistaken_ junior colleague:

- Much of the work involves finding and providing the right information to the agent
- With the right information, the agent is _extremely helpful_—this is especially true in the original research process and in identifying areas of code to fix
- The agent can deliver mostly working code, but you are the ultimate reviewer and your judgment is crucial
- There might be times when you should directly intervene

There was a good discussion recently about whether agents are making developers more productive. [Ethan Mollick](https://x.com/emollick/status/1943426233791869348) had a great summary of recent papers on this. In this case:

- Gemini was able to accelerate my investigation of a novel area, making a solution to my newsletter problem possible
- It also got _most_ of the way to putting a solution together
- But the long last mile ended up taking longer than the first parts of the project

That's quite telling. If I were experienced with this, had a good sense of what I wanted from the beginning, and was confident in my ability, using Gemini would have been worse than doing it myself. But that's the thing—I really don't know what I'm doing in web development, so Gemini was a huge help. It's hard to quantify exactly how much it sped up this change, since I doubt I would have been able to do much at all without assistance.

Many problems from this process come from Gemini itself:

- The code truncation problem in Code Assist, which I had to fix several times, is really annoying
- Gemini CLI is surprisingly slow compared to Claude Code
- That says nothing of its capabilities either—a better agent likely would have fixed my form styling issue without my intervention

So I can walk away from this both grateful that real work happens in Claude Code and excited about where all of this is going. It makes it easy to understand both sides of the [Vibe Coding](https://cloud.google.com/discover/what-is-vibe-coding?hl=en) hype and anti-hype cycle.

On one hand, 

- It has never been better to be a digital creator
- There are so many tools available to quickly bridge skill gaps and get something done

On the other hand, 

- You need good foundational knowledge and practices
- You need to be able to test and debug anything you develop with agents
- You need to understand context, and you need to be ready to deal with weird errors
- These are brand new products!

Thanks for reading all the way to the end, and I hope you enjoyed this guided tour of how I added one little box to the end of my blog posts. If you want to subscribe, that'd be pretty neat too!

## Tools Used

- The Gemini App helped me draft my plan
- [Gemini Code Assist](https://codeassist.google/) did the first round of edits to implement my project
- [Gemini CLI](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/) got almost to the end
- Claude was my [editor](https://claude.ai/share/67e53074-270a-49e7-9789-a025f205dd0e); he's responsible for all of the em-dashes above