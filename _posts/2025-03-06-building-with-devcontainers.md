---
layout: post
title: "Moving the development of this blog to devcontainers"
excerpt: "The world has changed since I discussed building the site for the first time. Let's get caught up!"
tags: [GitHub Sites, Devcontainers, Jekyll]
comments: true
modified: 2025-03-06
use_math: true
image: 
    feature: ski.png
    credit: George Arents Collection, The New York Public Library.
    creditlink: https://digitalcollections.nypl.org/items/510d47da-7947-a3d9-e040-e00a18064a99
---

{% include _toc.html %}

## Introduction

It's been nine years since I've worked on this blog. Time flies, and things sure have changed. Because of that, this seemed like a good time to update on how a static GitHub site like this can be maintained. This won't necessarily be a long post, more of those later, and it will cover the same big ideas as [my article on building the site](/how-i-built-this-site) for the first time.

What's changed?

*  We run in blogdown now
*  Everything needed for the site can run in a codespace; all editing is now in the browswer
*  Codespaces can be configured with a devcontainer, giving us one click to a fully running web environment
*  You can set up all of your dependencies to install automatically

That's what we'll cover today.

## Switching to blogdown

This site was originally built with `knitr` and `servr`, but the modern way to do this with R is to use [`blogdown`](https://github.com/rstudio/blogdown). `blogdown` renders RMarkdown files and then produces the static site using `hugo`. That clashes a little with tech stack powering GitHub pages, which still uses [jekyll](https://jekyllrb.com/). Fortunately, Yihui put together the scripts that you need to use `blogdown` and `jekyll` in this [repo](https://github.com/yihui/blogdown-jekyll). I've essentially copied them over, with a small modification to use `bundle`, following other Ruby practices. See below.

Either way, that was the first big change for the blog. I moved everything to blogdown, and my updated R scripts are in the `R/` directory.

## Configuring the codespace for GitHub Pages

[GitHub Codespaces](https://github.com/features/codespaces) are an extremely convenient option for setting up a dev environment. They are connected to your GitHub repo, and with a single button click you get an instance of VS Code running in your browser. Creating a codespace from a repo is very easy. Click the code button and then **Create a codespace**.

![codespace-from-repo](https://github.com/github/docs/raw/main/assets/images/help/codespaces/who-will-pay.png)

More details in the [quickstart guide](https://docs.github.com/en/codespaces/getting-started/quickstart).

A running instance of VS Code is already great, but it gets better. Codespaces are configurable from docker images. Your click can create a tailored development environment. Building might take a little time, but it removes almost all of the headaches of setting up a development environment. We need two basic things for our development environment:

*  An installation of R with `blogdown`
*  An installation of Ruby with `jekyll`

That doesn't really exist in the default [dev containers](https://github.com/devcontainers) offered by GitHub, but it's a solvable problem:

*  We'll start by installing the Rocker R container; which can be done in a runnning [VS Code instance](https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/introduction-to-dev-containers#using-a-predefined-dev-container-configuration)
*  We'll add Ruby and [Bundler](https://bundler.io/) as a 'feature' in the file
*  We'll get the rest of our dependencies installed in a post-creation step

Here's the complete devcontainer file:

```json
// For format details, see https://aka.ms/devcontainer.json. For config options, see the
// README at: https://github.com/rocker-org/devcontainer-templates/tree/main/src/r-ver
{
	"name": "R (rocker/r-ver base)",
	// Or use a Dockerfile or Docker Compose file. More info: https://containers.dev/guide/dockerfile
	"image": "ghcr.io/rocker-org/devcontainer/r-ver:4",
	"features": {
		"ghcr.io/devcontainers/features/ruby:1": {}
	},

	// Features to add to the dev container. More info: https://containers.dev/features.
	// "features": {},

	// Use 'forwardPorts' to make a list of ports inside the container available locally.
	// "forwardPorts": [],

	// Use 'postCreateCommand' to run commands after the container is created.
	"postCreateCommand": "bash .devcontainer/install_deps.sh"

	// Configure tool-specific properties.
	// "customizations": {},

	// Uncomment to connect as root instead. More info: https://aka.ms/dev-containers-non-root.
	// "remoteUser": "root"
}
```

And the bash script that it executes. Remember to always `chmod +x`.

```sh
#!/usr/bin/env bash

# Jekyll setup
bundle install
bundle add jekyll

# R Blogdown setup
Rscript -e "renv::install('blogdown')"
```

## Some gotchas, port forwarding and Jekyll

Once the dependencies are set up we should be able to render and serve the site from the terminal.

```sh
# If Rmd files have changes
Rscript -e "blogdown::build_site()"

# Of if you just want to build and serve with Jekyll
JEKYLL_ENV=production bundle exec jekyll build
JEKYLL_ENV=production bundle exec jekyll serve
```

VS Code will handle the port forwarding for you and give you a link to preview the blog.

My first attempts to get the preview up led to most of the page assets not appearing. The links were set to `http://localhost:PORT` but this was getting blocked by the browser. [Michael Rose](https://mademistakes.com/mastering-jekyll/site-url-baseurl/), the creator of the theme used for this site, mentions that this is a pretty common problem related to the `_config.yml` file used by Jekyll. The easiest fix is to:

*  Set the `JEKYLL_ENV` environment variable to avoid some default behaviors that aren't helpful
*  Define the url that we'll actually want

That last part depends on some [Codespaces behavior](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace#using-command-line-tools-and-rest-clients-to-access-ports). The URL that VS Code creates uses the following pattern: `https://CODESPACENAME-PORT.app.github.dev`. Since this is new each time a new codespace is created, I've been manually setting this in the YAML file once I'm ready to preview the page.