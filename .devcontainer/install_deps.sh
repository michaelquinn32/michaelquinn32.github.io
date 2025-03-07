#!/usr/bin/env bash

# Jekyll setup
bundle install
bundle add jekyll

# R Blogdown setup
Rscript -e "renv::install('blogdown')"
