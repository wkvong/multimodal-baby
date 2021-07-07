library(tidyverse)
library(ggthemes)

## read in logs
rnn_random_init_log <- read_csv("logs/multimodal_rnn_random_init/version_11/metrics.csv") %>%
    mutate(model = 'LSTM')
## rnn_pretrained_init_log <- read_csv("logs/multimodal_rnn_pretrained_init/version_2/metrics.csv") %>%
##     mutate(model = 'rnn_pretrained_init')
word_embed_random_init_log <- read_csv("logs/multimodal_word_embed_random_init/version_3/metrics.csv") %>%
    mutate(model = 'Embedding Only')

## combine logs
log <- bind_rows(list(rnn_random_init_log,
                      ## rnn_pretrained_init_log,
                      word_embed_random_init_log))

## plot train loss
train_loss_df <- log %>%
    group_by(model) %>%
    select(epoch, train_loss) %>%
    filter(!is.na(train_loss))

ggplot(train_loss_df, aes(x = epoch, y = train_loss, color = model)) +
    geom_line() +
    ggtitle("training loss")

## plot validation loss/accuracy
validation_accuracy_df <- log %>%
    group_by(model) %>%
    select(epoch, val_loss) %>%
    filter(!is.na(val_loss)) %>%
    mutate(val_accuracy = 1 - val_loss)

ggplot(validation_accuracy_df, aes(x = epoch, y = val_accuracy, color = model)) +
    stat_smooth(aes(x = epoch, y = val_accuracy), method = "loess") +
    geom_hline(yintercept = 0.25, linetype='dashed') +
    geom_line() +
    ggtitle("validation accuracy") +
    ylim(0, 0.6)

## read in validation results
rnn_random_init_val_results <- read_csv("results/multimodal_rnn_random_init_val_results.csv") %>%
    mutate(model = 'LSTM', `Word for Cat`='cat')    
## rnn_pretrained_init_val_results <- read_csv("results/multimodal_rnn_pretrained_init_val_results.csv") %>%
##     mutate(model = 'rnn_pretrained_init', cat_word='cat')
word_embed_random_init_val_results <- read_csv("results/multimodal_word_embed_random_init_val_results.csv") %>%
    mutate(model = 'Embedding', `Word for Cat`='cat')

val_results <- bind_rows(list(rnn_random_init_val_results,
                              ## rnn_pretrained_init_val_results,
                              word_embed_random_init_val_results))

## plot overall results
val_summary <- val_results %>%
    group_by(model) %>%
    summarise(accuracy_mean = mean(correct),
              accuracy_se = sd(correct) / sqrt(n()))

ggplot(val_summary, aes(x = model, y = accuracy_mean, fill = model)) +
    geom_bar(stat="identity", position=position_dodge()) +
    geom_errorbar(aes(ymin = accuracy_mean - accuracy_se, ymax = accuracy_mean + accuracy_se),
                  position=position_dodge(0.9), width = 0.25) +
    geom_hline(yintercept=0.25, linetype='dashed') +
    xlab("Model") +
    ylab("Evaluation accuracy") +
    ylim(0, 0.75) +
    theme(legend.position = "none",
          ## panel.border = element_rect(colour = "black", size = 1),
          axis.title.x = element_text(size = 20),
          axis.title.y = element_text(size = 20),
          axis.text.x = element_text(size = 16),
          axis.text.y = element_text(size = 16),
          axis.ticks.x = element_line(colour = "black"),
          axis.ticks.y = element_line(colour = "black"),
          legend.title = element_text(size = 16),
          legend.text = element_text(size = 16),
          strip.text.x = element_text(size = 20),
          strip.text.y = element_text(size = 20))

## plot results by category
val_summary_by_category <- val_results %>%
    group_by(model, target_category) %>%
    summarise(accuracy_mean = mean(correct),
              accuracy_se = sd(correct) / sqrt(n())) %>%
    mutate(Model = model)

ggplot(val_summary_by_category, aes(x = target_category, y = accuracy_mean, fill = Model)) +
    geom_bar(stat="identity", position=position_dodge()) +
    geom_errorbar(aes(ymin = accuracy_mean - accuracy_se, ymax = accuracy_mean + accuracy_se),
                  position=position_dodge(0.9), width = 0.25) +
    geom_hline(yintercept=0.25, linetype='dashed') +
    xlab("Evaluation categories") +
    ylab("Evaluation accuracy") +
    theme(legend.position = "none",
          axis.title.x = element_text(size = 20),
          axis.title.y = element_text(size = 20),
          axis.text.x = element_text(size = 16),
          axis.text.y = element_text(size = 16),
          axis.ticks.x = element_line(colour = "black"),
          axis.ticks.y = element_line(colour = "black"),
          legend.title = element_text(size = 16),
          legend.text = element_text(size = 16),
          strip.text.x = element_text(size = 20),
          strip.text.y = element_text(size = 20))

## compare kitty results
rnn_random_init_val_kitty_results <- read_csv("results/multimodal_rnn_random_init_val_kitty_results.csv") %>%
    mutate(model = 'LSTM', `Word for Cat`='kitty')    
## rnn_pretrained_init_val_kitty_results <- read_csv("results/multimodal_rnn_pretrained_init_val_kitty_results.csv") %>%
##     mutate(model = 'rnn_pretrained_init', cat_word='kitty')
word_embed_random_init_val_kitty_results <- read_csv("results/multimodal_word_embed_random_init_val_kitty_results.csv") %>%
    mutate(model = 'Embedding', `Word for Cat`='kitty')

kitty_results <- bind_rows(list(rnn_random_init_val_results,
                                ## rnn_pretrained_init_val_results,
                                word_embed_random_init_val_results,
                                rnn_random_init_val_kitty_results,
                                ## rnn_pretrained_init_val_kitty_results,
                                word_embed_random_init_val_kitty_results))

kitty_summary <- kitty_results %>%
    group_by(model, `Word for Cat`, target_category) %>%
    summarise(accuracy_mean = mean(correct),
              accuracy_se = sd(correct) / sqrt(n())) %>%
    filter(target_category == 'cat')

ggplot(kitty_summary, aes(x = model, y = accuracy_mean, fill = `Word for Cat`)) +
    geom_bar(stat="identity", position=position_dodge()) +
    geom_errorbar(aes(ymin = accuracy_mean - accuracy_se, ymax = accuracy_mean + accuracy_se),
                  position=position_dodge(0.9), width = 0.25) +
    geom_hline(yintercept=0.25, linetype='dashed') +
    xlab("Evaluation categories") +
    ylab("Evaluation accuracy") +
    theme(legend.position = "bottom",
          axis.title.x = element_text(size = 20),
          axis.title.y = element_text(size = 20),
          axis.text.x = element_text(size = 16),
          axis.text.y = element_text(size = 16),
          axis.ticks.x = element_line(colour = "black"),
          axis.ticks.y = element_line(colour = "black"),
          legend.title = element_text(size = 16),
          legend.text = element_text(size = 16),
          strip.text.x = element_text(size = 20),
          strip.text.y = element_text(size = 20))

## descriptives
training_data <- read_csv("data/frame_utterance_pairs_clean.csv")

eval_categories <- c("ball", "basket", "car", "cat", "chair", "computer", 
                     "crib", "door", "floor", "foot", "ground", "hand", 
                     "kitchen", "paper", "puzzle", "road", "room", "sand", 
                     "stairs", "table", "toy", "window")

# calculate frequencies of each category
eval_category_freq <- list()
for (i in 1:length(eval_categories)) {
    # initialize counts
    eval_category <- eval_categories[i]
    eval_category_freq[eval_category] = 0
}

word_frequency <- training_data$utterance %>%
    strsplit(split = ' ') %>%
    unlist() %>%
    table() %>%
    sort(decreasing = TRUE)

## filer for eval categories and append to val summary df
eval_category_frequency <- word_frequency[eval_categories]

val_summary_by_category <- val_summary_by_category %>%
    mutate(word_freq = eval_category_frequency)

## plot correlation
ggplot(val_summary_by_category, aes(x = log(word_freq), y = accuracy_mean)) +
    geom_point() +
    geom_smooth(method='lm', formula=y~x) +
    facet_grid(~ model) +
    xlab("Log(Word Frequency)") +
    ylab("Evaluation accuracy") +
    theme(legend.position = "bottom",
          axis.title.x = element_text(size = 20),
          axis.title.y = element_text(size = 20),
          axis.text.x = element_text(size = 16),
          axis.text.y = element_text(size = 16),
          axis.ticks.x = element_line(colour = "black"),
          axis.ticks.y = element_line(colour = "black"),
          legend.title = element_text(size = 16),
          legend.text = element_text(size = 16),
          strip.text.x = element_text(size = 20),
          strip.text.y = element_text(size = 20))
    
embedding <- val_summary_by_category %>% filter(model == "Embedding")
cor(embedding$accuracy_mean, log(embedding$word_freq))^2

lstm <- val_summary_by_category %>% filter(model == "LSTM")
cor(lstm$accuracy_mean, log(lstm$word_freq))^2
