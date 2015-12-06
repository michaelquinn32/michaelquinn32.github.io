# Implementing Spam experiment

## ---- prelims
# Prelims -----------------------------------------------------------------

library(dplyr)
library(purrr)
library(readr)
library(knitr)
library(ggplot2)
library(caret)
library(adventureR)

options(digits = 4)

## ---- lda_example
# LDA example -------------------------------------------------------------

mus <- list(group1 = c(0,0), group2 = c(2,2))
Sigma <- matrix(c(2, -1, -1, 2), nrow = 2)

example_data <- map(mus, ~ MASS::mvrnorm(n = 50, mu = .x, Sigma = Sigma)) %>% 
    map(as.data.frame) %>%
    bind_rows(.id = "group")

example_lda <- LDA(group ~ . , example_data)

slope <- -1 * reduce(example_lda$a, `/`)
intercept <- -1 * example_lda$b %>% as.numeric %>% `/`(example_lda$a[2])

ggplot(example_data, aes(V1, V2)) +
    geom_point(aes(color = group)) +
    geom_abline(slope = slope, intercept = intercept, lty = 2) +
    scale_color_brewer(palette = "Set1") +
    ggtitle("LDA Example")


## ---- data
# Links -------------------------------------------------------------------

spam_http <- "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
spam_names <- "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"


# Get Data ----------------------------------------------------------------

col_names <- read_lines(spam_names, skip = 33) %>% gsub(":.*", "", .) %>% c("target")
col_types <- c(replicate(55, col_double()), replicate(3, col_integer()))
spam_data <- read_csv(spam_http, col_names = col_names, col_types = col_types)


# Fix target --------------------------------------------------------------

spam_data <- mutate(spam_data, target = as.factor(target))

## ---- eda_x
# Basic EDA ---------------------------------------------------------------

summary_funs <- list(mean = partial(mean, na.rm = TRUE),
                     sd = partial(sd, na.rm = TRUE),
                     min = partial(min, na.rm = TRUE),
                     p25 = partial(quantile, probs = .25, names = FALSE, na.rm = TRUE),
                     median = partial(median, na.rm = TRUE),
                     p75 = partial(quantile, probs = .75, names = FALSE, na.rm = TRUE),
                     max = partial(max, na.rm = TRUE))

spam_data[-58] %>% map(~ invoke_map(summary_funs, x = .x)) %>%
    invoke(rbind.data.frame, .) %>%
    kable

## ---- eda_y
spam_data %>% count(target) 

## ---- test_models
# Implementing the Experiment ---------------------------------------------

# Store models to be tested
TC <- trainControl(method = 'none')
NB_pars <- data.frame(fL = 0, usekernel = FALSE)
proc <- c("nzv")

models <- list(my_lda = partial(LDA, target ~ .),
               caret_lda = partial(train, target ~ ., method = 'lda',
                                   trControl = TC),
               caret_qda = partial(train, target ~ ., method = 'qda',
                                   trControl = TC, preProcess = proc),
               my_nb = partial(naive_bayes, target ~ .),
               caret_nb = partial(train, target ~ ., method = 'nb',
                                  trControl = TC, tuneGrid = NB_pars,
                                  preProcess = proc))

## ---- random_splits
# Create list of random training and test dataframes

dfs <- rerun(100, split_data(spam_data))

## ---- test_dr
# Save results so don't need to compute again and again
if (file.exists("../post-elements/2015-12-02-spam-results.rds")) {
    results <- readRDS("../post-elements/2015-12-02-spam-results.rds")

} else {
    # Fit and score models
    results <- dfs %>%
        map( ~ invoke_map(models, data = .x$train)) %>%
        map2(dfs, ~ map(.x, score_model, newdata = .y$test,
                        actual = .y$test$target))

    # Save results
    saveRDS(results, "../post-elements/2015-12-02-spam-results.rds")
}

## ---- final_plot
# Lets make a nice plot

results %>% at_depth(2, data.frame) %>%
    at_depth(1, bind_rows, .id = "model") %>%
    bind_rows(.id = "iteration") %>%
    ggplot(aes(model, f1)) +
    geom_boxplot() +
    ggtitle("Comparison of F1 in classification models")

