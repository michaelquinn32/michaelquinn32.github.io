# From: https://github.com/yihui/blogdown-jekyll/blob/gh-pages/R/build_one.R

local({
    # fall back on '/' if baseurl is not specified
    baseurl = servr:::jekyll_config('.', 'baseurl', '/')
    knitr::opts_knit$set(base.url = baseurl)
    # fall back on 'kramdown' if markdown engine is not specified
    markdown = servr:::jekyll_config('.', 'markdown', 'kramdown')
    # see if we need to use the Jekyll render in knitr
    if (markdown == 'kramdown') {
        knitr::render_jekyll()
    } else knitr::render_markdown()
    
    # input/output filenames are passed as two additional arguments to Rscript
    a = commandArgs(TRUE)
    d = gsub('^_|[.][a-zA-Z]+$', '', a[1])

    # set where you want to host the figures (I store them in my Dropbox Public
    knitr::opts_chunk$set(fig.path = sprintf('images/%s/', gsub('^.+/', '', d)),
                          cache.path = '_cache/',
                          cache = T,
                          fig.width = 8,
                          fig.height = 5,
                          dpi = 150,
                          out.width = '800px',
                          out.height = '500px')
    knitr::opts_knit$set(base.dir = '~/Projects/michaelquinn32.github.io/',
                         base.url = 'https://michaelquinn32.github.io/')
    
    knitr::opts_knit$set(width = 70)
    knitr::knit(a[1], a[2], quiet = TRUE, encoding = 'UTF-8', envir = .GlobalEnv)
})
