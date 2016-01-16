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
    mutate(bias.lwr = (lwr - true$lwr), bias.upr = upr - true$upr,
           bias.sqr = bias.lwr^2 + bias.upr^2)

knitr::kable(tests[-5], digits = 4)

## ---- experiment

# Simulate replacement bias -----------------------------------------------

test <- bootstrap_experiment(red_funs, length.out = 2, niter = 2)


## ---- save_test

saveRDS(test, "post-elements/2016-01-03-bootstrap.rds")


## ---- import_test

test <- readRDS("post-elements/2016-01-03-bootstrap.rds")


## ---- box_plot

# Box plot
test %>%
    gather(bias_type, value, bias.lwr, bias.upr, bias.sqr) %>%
    ggplot(aes(method, value)) +
    geom_boxplot() +
    facet_grid(replace ~ bias_type, scales = 'free') +
    ggtitle('Box plot of biases for each method')


## ---- summary

# Simple summary statistics by group
test %>% 
    group_by(replace, method) %>% 
    summarize(
        lwr.mean = mean(bias.lwr),
        lwr.sd = sd(bias.lwr),
        lwr.abs.min = min(abs(bias.lwr)),
        lwr.abs.med = median(abs(bias.lwr)),
        lwr.abs.max = max(abs(bias.lwr))
    ) %>% 
    knitr::kable(., digits = 4)

## ---- t1

# T test: Is the SE error different from 0?
test %>% 
    group_by(replace, method) %>% 
    summarize(
        lwr.stat = t.test(bias.lwr)$statistic,
        lwr.pvalue = t.test(bias.lwr)$p.value
    ) %>% 
    knitr::kable(., digits = 4)


## ---- anova

# Anova, convert test levels to factors
reshape <- test %>%
    filter(replace == FALSE) %>% 
    select(bias.lwr, method, n, r, mu, sigma, bsi) %>%
    mutate(bias.lwr = abs(bias.lwr), 
           method = factor(method, levels = c("ss_qntls", "emp_qntls", "se"))) 

model <- aov(bias.lwr ~ (.) * (.) , reshape)
summary(model)

## ---- updated
model <- aov(bias.lwr ~ method + n + r + sigma + bsi + n:r + n:sigma + r:sigma, reshape)
summary(model)

## ---- plot_fun

model %>% model.matrix %>% 
    as.data.frame %>% 
    select(4:7) %>% 
    mutate(fitted =  fitted(model)) %>% 
    gather(variable, predictor, -fitted) %>% 
    group_by(predictor) %>% 
    mutate(av_fitted = mean(fitted)) %>%
    ggplot(aes(predictor)) +
    facet_wrap(~ variable, scales = 'free') +
    geom_point(aes(y = fitted, color = variable)) +
    geom_line(aes(y = av_fitted), color = "black") +
    theme(legend.position = "none") +
    ggtitle('Fitted values against predictors')


## ---- hierarchical_model

model_hier <- lmer(bias.lwr ~ r + I(r^2) + (1 + r + I(r^2) | n), reshape)

quad_mod <- function(row, x) {
    drop(cbind(1, x, x^2) %*% row)
}

# Generate fitted values for hier model
multi_fits <- function(model, FUN, x) {
    coef_df <- coef(model)
    nms <- rownames(coef_df[[1]])
    fits <- map(transpose(coef_df[[1]], .simplify = TRUE), FUN, x)
    as_data_frame(set_names(fits, nms))
}

plot_df <- function(model, FUN, vars){
    frame <- model.frame(model)
    target <- set_names(frame[1], "target")
    predictors <- frame[vars]
    fits <- map(as.list(predictors), ~ multi_fits(model, FUN, .x))
    cmbd <- bind_cols(fits)
    gather_(cbind(target, bind_cols(predictors), cmbd), "level", "value", names(cmbd))
}

plot_df(model_hier, quad_mod, "r") %>% 
    ggplot(aes(r)) +
    geom_point(aes(y = target), color = "black", alpha = .4) +
    geom_line(aes(y = value, color = level)) +
    ggtitle("Abs bias, lower tail, vs subsampling rate, by n")
