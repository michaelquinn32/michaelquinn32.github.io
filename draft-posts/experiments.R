# Coin Flip Bayes


# Model Functions ---------------------------------------------------------

beta_expectation <- function(a, b) {
    a / ( a + b)
}

beta_variance <- function(a, b) {
    a * b / ((a + b)^2 * (a + b + 1))
}

int_length <- function(a, b, u, prob) {
    u - qbeta(pbeta(u, a, b) - prob, a, b)
}

credible_interval <- function(a, b, prob, range = c(0,1)) {
    suppressWarnings(opt <- optimize(int_length, range, a = a, b = b, prob = prob))
    list(upper = opt$minimum, lower = opt$minimum - opt$objective)
}


# Experiments -------------------------------------------------------------

library(ggplot2)
library(purrr)
library(dplyr)

alpha <- 2
beta <- 2
successes <- 560
attempts <- 1000

posterior_params <- function(alpha, beta, success, attempts) {
    list(a = alpha + success, b = beta + attempts - success)
}

test_summary <- list(mean = beta_expectation,
                     variance = beta_variance,
                     interval = partial(credible_interval, prob = .95))


results <- test_summary %>% 
    invoke_map(list(posterior_params(alpha, beta, successes, attempts)))

list(steps = c(.45, .65), 
     interval = c(results$interval$lower, results$interval$upper)) %>% 
    map(~ seq(.x[1], .x[2], length.out = 1000)) %>% 
    data.frame(dist = map(., ~ dbeta(.x, alpha + success, beta + attempts - success))) %>% 
    ggplot() + 
        geom_line(aes(x = steps, y = dist.steps)) +
        geom_area(aes(x = interval, y = dist.interval), 
                  fill = "purple", alpha = .5, color = "black") +
        annotate("text", .45, 22.5, size = 4, hjust = 0,
                label = paste("Mean =",round(results$mean, 4),
                      "\nVariance =", round(results$variance, 4),
                      "\nLower =", round(results$interval$lower, 4), 
                      "\nUpper =", round(results$interval$upper, 4))) +
        ggtitle(paste("Minimum length credible interval\nCoin flip experiment"))
        


# Experiment 2 ------------------------------------------------------------

inv_cdf <- function(u, b, lambda, n) {
    -1/ lambda * log(1 + u^(1/n) * (exp(-lambda * b) - 1))
}

rerun(1000, runif(1000)) %>%
    map_dbl(~5 - mean(inv_cdf(.x, 5, 1/4, 500))) %>% 
    data_frame %>% 
    set_names("test") %>%  
    ggplot(aes(x = test)) + geom_density()
     
