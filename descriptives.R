library(tidyverse)
library(tidytext)
library(ggthemes)

df <- read_csv("data/frame_utterance_pairs_clean.csv")

df <- df %>%
    mutate(len = str_count(utterance, '\\w+'))

df %>%
    filter(len < 50) %>%
    ggplot(aes(x = len)) +
    geom_bar() +
    xlab("Utterance Length") +
    ylab("Frequency")

eval_words <- c("ball", "cat", "crib", "foot", "kitchen", "road", "stairs", "window", "basket", "chair", "door", "ground", "paper", "room", "table", "car", "computer", "floor", "hand", "puzzle", "sand", "toy")

eval_words_df <- df %>%
    unnest_tokens(word, utterance, token = "words") %>%
    filter(word %in% eval_words) %>%
    group_by(word) %>%
    summarise(n = n())

eval_words_df %>%
    ggplot(aes(x = word, y = n, fill = word)) +
    geom_bar(stat="identity") +
    coord_flip() +
    theme(legend.position = "none") +
    xlab("Word") +
    ylab("Frequency")
    
    
