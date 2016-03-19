#' Experiments in Bootstrapping
#' Michael Quinn
#'
#' The script below implement four different sampling algorithms to derive the
#' confidence intervals for the mean of a value. To start, I show the traditional
#' bootstrap, which resamples the full vector with replacement.
#'
#' This is compared to a subset resampler. There are three different methods
#' illustrated. The first shows how a subset resampler is biased. Using some
#' corrections availabe in research, I fix this. Sub-sampling is also supposed to work
#' with samples without replacement, but the examples here seem to show that it converges
#' slower.

## ---- prelims

library(purrr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(parallel)
library(adventureR)

## ---- comparison

# Comparing BS sampling schemes -------------------------------------------

n <- 1000
b <- n * .25
x <- rnorm(n, 4, 2)
mu <- mean(x)


# Actual values
se <- sd(x) / sqrt(n)
true <- c(lwr = .025, upr = .975) %>% map(qnorm, mean(x), se)

# CI functions
full_funs <- list(se = se_ci, qntls = qntl)
red_funs <- list(se = red_se, ss_qntls = ss_quantiles, emp_qntls = quantile_ev)

# Bootstrapped values
full <- bs_sim(x, 10000, replace = TRUE, full_funs, n, b = n, mu)
biased <- bs_sim(x, 10000, replace = TRUE, full_funs, n, b, mu)
red_rep <- bs_sim(x, 10000, replace = TRUE, red_funs, n, b, mu)
red_nrep <- bs_sim(x, 10000, replace = FALSE, red_funs, n, b, mu)


## ---- results

# Test values
tests <- list(full, biased, red_rep, red_nrep) %>%
    set_names(nm = c("full", "biased", "adjusted, rep", "adjusted, no rep")) %>% 
    bind_rows(.id = "type") %>%
    mutate(bias.lwr = (lwr - true$lwr), 
           bias.upr = upr - true$upr,
           bias.sqr = sqrt(bias.lwr^2 + bias.upr^2))

knitr::kable(tests[-5], digits = 4)


## ---- convergence

# Create a range of iterations
i_range <- cross_n(list(x = c(5,10), y = 1:5)) %>% 
    map_dbl(~ .$x^.$y)

# Test errors across i_range
full_err <- map(i_range, bs_sim, x = x, replace = TRUE, full_funs, n, b = n, mu)
subsamp_err <- map(i_range, bs_sim, x = x, replace = FALSE, red_funs[c(1,2)], n, b = b, mu)
converge_results <- list(standard = full_err, subsamp = subsamp_err) %>% 
    map(bind_rows) %>% 
    bind_rows(.id = "sampling") %>% 
    mutate(method = gsub("^.*_", "", method),
           bias.sqr = sqrt((true$lwr - lwr)^2 + (true$upr - upr)^2),
           iter = rep(i_range, 2, each = 2))

# Plot
converge_results %>% 
    ggplot(aes(iter, bias.sqr)) +
    geom_line(aes(color = sampling)) +
    facet_grid(. ~ method) +
    scale_x_log10() +
    scale_y_log10() +
    ggtitle('Convergence Rates for Standard Bootstrap and Subsampling')

## ---- experiment

# Simulate replacement bias -----------------------------------------------

test <- bootstrap_experiment(red_funs)


## ---- save_test

saveRDS(test, "post-elements/2016-01-17-bootstrap.rds")


## ---- import_test

test <- readRDS("../post-elements/2016-01-17-bootstrap.rds")


## ---- box_plot

# Box plot
test %>%
    gather(bias_type, value, bias.lwr, bias.upr, bias.sqr) %>%
    ggplot(aes(method, value)) +
    geom_boxplot(outlier.shape =  1) +
    facet_grid(replace ~ bias_type, scales = 'free') +
    ggtitle('Box plot of biases for each method')


## ---- summary

# Simple summary statistics by group, square
test %>% 
    group_by(replace, method) %>% 
    summarize(
        mean = mean(bias.sqr),
        sd = sd(bias.sqr),
        min = min(bias.sqr),
        med = median(bias.sqr),
        max = max(bias.sqr)
    ) %>% 
    knitr::kable(., digits = 4)

## ---- t1

# T test: Is the SE error different from 0?
test %>% 
    group_by(replace, method) %>% 
    summarize(
        stat = t.test(bias.sqr)$statistic,
        pvalue = t.test(bias.sqr)$p.value
    ) %>% 
    knitr::kable(., digits = 4)


## ---- anova

# Anova, convert test levels to factors
reshape <- test %>%
    select(bias.sqr, method, n, r, sigma, bsi, replace) %>%
    mutate(method = factor(method, levels = c("ss_qntls", "emp_qntls", "se"))) 

model <- aov(bias.sqr ~ (.) * (.) - method:r - r:bsi, reshape)
MASS::boxcox(model)

# Show boxcox of original model
model <- update(model, I((bias.sqr)^(1/3)) ~ .)

## ---- residuals
# Plot residuals
data.frame(residuals = rstandard(model),
           fitted = model.frame(model)[[1]]) %>% 
    ggplot(aes(fitted, residuals)) +
    geom_point(alpha = .25) +
    ggtitle('Fitted vs standardized residuals')

## ---- effect
tables <- model.tables(model, "effects")
tables$tables$`r:replace`

## ---- plot_fun

model %>% model.matrix %>% 
    as.data.frame %>% 
    select(4:8) %>% 
    mutate(fitted =  fitted(model)^2) %>% 
    gather(variable, predictor, -fitted, -replaceTRUE) %>% 
    rename(replace = replaceTRUE) %>% 
    mutate(replace = as.factor(ifelse(replace == 1, "true", "false"))) %>% 
    group_by(predictor, replace) %>% 
    mutate(av_fitted = mean(fitted)) %>% 
    ggplot(aes(predictor)) +
    facet_grid(replace ~ variable, scales = 'free') +
    geom_point(aes(y = fitted, color = variable)) +
    geom_line(aes(y = av_fitted), color = "black") +
    theme(legend.position = "none") +
    ggtitle('Fitted values against predictors')
