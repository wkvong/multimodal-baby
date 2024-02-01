library(tidyverse)
library(scales)
library(ggrepel)
library(RColorBrewer)
library(glue)
library(patchwork)

setwd('~/code/multimodal-baby/analysis')

## read summary bounds csv
saycam_summary <- read_csv("../results/summary/saycam-bounds-summary.csv")

## extract configs for summary bounds figure
saycam_bounds <- saycam_summary %>%
    filter(config == "contrastive_embedding" & filtered == FALSE | config == "contrastive_shuffled" |
               config == "supervised_linear_probe_all" | config == "clip (vit-l/14)")

## also add frozen random init as another lower bound
saycam_frozen_random_init <- read_csv("../results/summary/saycam-ablations.csv") %>%
    filter(config == "contrastive_embedding_frozen_random_init") 

## rename configs and re-order
saycam_bounds <- saycam_bounds %>%
    select(-X1)
    
# combine 
saycam_bounds <- bind_rows(saycam_bounds, saycam_frozen_random_init) 
saycam_bounds <- saycam_bounds %>%
    mutate(config = ifelse(config == "contrastive_embedding", "CVCL", config)) %>%
    mutate(config = ifelse(config == "contrastive_shuffled", "CVCL (Shuffled)", config)) %>%
    mutate(config = ifelse(config == "clip (vit-l/14)", "CLIP", config)) %>%
    mutate(config = ifelse(config == "supervised_linear_probe_all", "Linear Probe", config)) %>%
    mutate(config = ifelse(config == "contrastive_embedding_frozen_random_init", "CVCL (Rand. Features)", config)) %>%
    mutate(config = factor(config, levels = c("CVCL", "CVCL (Shuffled)", "CVCL (Rand. Features)", "CLIP", "Linear Probe")))

## get summary stats for main bounds
saycam_bounds_summary <- saycam_bounds %>%
    group_by(config, seed) %>%
    summarise(correct = mean(correct)) %>%
    group_by(config) %>%
    summarise(mean = 100*mean(correct), se = 100*sd(correct) / sqrt(n()))
saycam_bounds_summary

## bar plot of bounds, fig 2a
ggplot(saycam_bounds_summary, aes(x = config, y = mean, fill=config)) +
    geom_bar(stat = "identity") +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
    geom_hline(yintercept = 25, linetype = "dashed") +
    geom_text(aes(label=format(round(mean, digits=1), nsmall=1), y=0), position=position_dodge(width=0.9), size = 8, vjust=-0.25) +
    # theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    # scale_fill_brewer(palette="Spectral") +
    scale_fill_manual(values = c("#33A02C", "#FED976", "#FDBF6F", "#A6CEE3", "#1F78B4")) +
    labs(x = "Model", y = "Classification Accuracy") +
    scale_x_discrete(labels = function(x) 
        stringr::str_wrap(x, width = 6)) +
    ylim(0, 100) + 
    theme_bw(base_size=30) +
    theme(legend.position = "none",
          axis.text.x = element_text(size=20))

## save figure
## ggsave("../figures/saycam-bounds-summary.pdf", height=7, width=7, units="in", dpi=500) 

## results by target category
## get summary stats for main bounds
saycam_bounds_by_target_category_summary <- saycam_bounds %>%
    #filter(config != "CVCL (Shuffled)") %>%
    group_by(target_category, config, seed) %>%
    summarise(correct = mean(correct)) %>%
    group_by(target_category, config) %>%
    summarise(mean = 100*mean(correct), se = 100*sd(correct) / sqrt(n()))
saycam_bounds_by_target_category_summary

## get factor ordering based on contrastive model accuracy
target_category_factor_ordering <- saycam_bounds_by_target_category_summary %>%
    filter(config == "CVCL") %>%
    arrange(-mean) %>%
    pull(target_category)

## re-order factors
saycam_bounds_by_target_category_summary$target_category <- factor(
    saycam_bounds_by_target_category_summary$target_category, levels=target_category_factor_ordering)

## create new column for extended model name
saycam_bounds_by_target_category_summary <- saycam_bounds_by_target_category_summary %>%
    mutate(config_long = config) %>%
    mutate(config_long = case_when(
               config == "CVCL (Rand. Features)" ~ "CVCL (Random Features)"
           ))

capitalize <- function(string) {
    substr(string, 1, 1) <- toupper(substr(string, 1, 1))
    string
}

## fig 2d
ggplot(saycam_bounds_by_target_category_summary, aes(x = config, y = mean, fill = config)) +
    geom_bar(stat = "identity") +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
    geom_hline(yintercept = 25, linetype = "dashed") +
    scale_fill_manual(values = c("#33A02C", "#FED976", "#FDBF6F", "#A6CEE3", "#1F78B4"), name = "Model", labels = c("CVCL", "CVCL (Shuffled)", "CVCL (Random Features)", "CLIP", "Linear Probe")) +
    labs(x = "Model", y = "Classification Accuracy") +
    scale_x_discrete(labels = function(x) 
        stringr::str_wrap(x, width = 10)) +
    ylim(0, 100) + 
    theme_bw(base_size=30) +
    theme(legend.position = "bottom",
          legend.title=element_text(size=24),
    #axis.text.x = element_text(angle = 90, hjust = 1)) +
    axis.text.x=element_blank(),
    axis.ticks.x=element_blank()) +
    facet_wrap(~ target_category, labeller = labeller(target_category = capitalize), ncol = 6)

## ggsave("../figures/saycam-bounds-by-target-category-summary.pdf", height=14, width=20, units="in", dpi=500) 

## do comparison between CVCL and linear probe
saycam_cvc_vs_linear_probe <- saycam_bounds_by_target_category_summary %>%
    filter(config == "CVCL" | config == "Linear Probe") %>%
    select(config, target_category, mean) %>%
    spread(config, mean) %>%
    mutate(diff = `Linear Probe` - CVCL)

## plot difference
ggplot(saycam_cvc_vs_linear_probe, aes(x = diff, y = target_category)) +
    geom_bar(stat = "identity") +
    labs(x = "Linear Probe - CVCL", y = "Target Category") +
    ## reverse ordering of y-axis
    scale_y_discrete(limits = rev(saycam_cvc_vs_linear_probe$target_category),
                     labels = function(x) 
                         stringr::str_wrap(x, width = 10)) +
    theme_bw(base_size=20) +
    theme(legend.position = "none")

## perform rank correlation
saycam_rank_cor <- cor.test(saycam_cvc_vs_linear_probe$CVCL,
                            saycam_cvc_vs_linear_probe$`Linear Probe`,
                            method="spearman")

## do comparison between unfiltered and filtered evaluation trials
saycam_filtered_comp <- saycam_summary %>%
    filter(config == "contrastive_embedding") %>%
    group_by(filtered, target_category, seed) %>%
    summarise(correct = mean(correct)) %>%
    group_by(filtered, target_category) %>%
    summarise(mean = 100*mean(correct), se = 100*sd(correct) / sqrt(n())) %>%
    ungroup() %>%
    group_by(target_category) %>%
    filter(n() > 1)

saycam_filtered_comp$Condition <- ifelse(saycam_filtered_comp$filtered, "Filtered", "Original")
saycam_filtered_comp$Condition <- factor(saycam_filtered_comp$Condition, levels=c("Original", "Filtered"))

## fig s3
ggplot(saycam_filtered_comp, aes(x = target_category, y = mean, group = Condition, fill = Condition)) +
    geom_bar(stat="identity", position=position_dodge()) +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2,
                  position=position_dodge(1)) +
    scale_fill_manual(values = c("#33A02C", "#88D78F")) +
    labs(x = "Target Category", y = "Classification Accuracy") +
    scale_x_discrete(labels = function(x) 
        stringr::str_wrap(capitalize(x), width = 6)) +
    ylim(0, 100) +
    theme_bw(base_size=30) +
    theme(legend.position = "bottom",
          axis.text.x = element_text(size=20, angle=45, hjust=1))
## ggsave("../figures/labeled-s-manual-filtering.pdf", height=10, width=12, units="in", dpi=500) 

## extract configs for linear probe summary
saycam_linear_probes <- saycam_summary %>%
    filter(config == "contrastive_embedding" & filtered == FALSE | config == "supervised_linear_probe_all" | 
           config == "supervised_linear_probe_1_percent" | config == "supervised_linear_probe_10_percent")

## rename configs and re-order
saycam_linear_probes <- saycam_linear_probes %>%
    mutate(config = ifelse(config == "contrastive_embedding", "CVCL", config)) %>%
    mutate(config = ifelse(config == "supervised_linear_probe_all", "Linear Probe (100%)", config)) %>%
    mutate(config = ifelse(config == "supervised_linear_probe_10_percent", "Linear Probe (10%)", config)) %>%
    mutate(config = ifelse(config == "supervised_linear_probe_1_percent", "Linear Probe (1%)", config)) %>%
    mutate(config = factor(config, levels = c("CVCL", "Linear Probe (1%)", "Linear Probe (10%)", "Linear Probe (100%)")))

## get summary stats for linear probes
saycam_linear_probes_summary <- saycam_linear_probes %>%
    group_by(config, seed) %>%
    summarise(correct = mean(correct)) %>%
    group_by(config) %>%
    summarise(mean = 100*mean(correct), se = 100*sd(correct) / sqrt(n()))
saycam_linear_probes_summary

## bar plot of linear probes, fig 2b
ggplot(saycam_linear_probes_summary, aes(x = config, y = mean, fill=config)) +
    geom_bar(stat = "identity") +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
    geom_text(aes(label=format(round(mean, digits=1), nsmall=1), y=0), position=position_dodge(width=0.9), size = 8, vjust=-0.25) +
    geom_hline(yintercept = 25, linetype = "dashed") +
    labs(x = "Model", y = "Classification Accuracy") +
    scale_x_discrete(labels = function(x) 
        stringr::str_wrap(x, width = 8)) +
    scale_fill_manual(values = c("#33A02C", "#BDD7E7", "#6BAED6", "#1F78B4")) +
    ylim(0, 100) + 
    theme_bw(base_size=30) +
    theme(legend.position = "none",
          axis.text.x = element_text(size=20))

## save figure
## ggsave("../figures/saycam-linear-probes-summary.pdf", height=7, width=7, units="in", dpi=500) 

## results by target category
## get summary stats for main bounds
saycam_linear_probes_by_target_category_summary <- saycam_linear_probes %>%
    filter(config != "CVCL (Shuffled)") %>%
    group_by(target_category, config, seed) %>%
    summarise(correct = mean(correct)) %>%
    group_by(target_category, config) %>%
    summarise(mean = 100*mean(correct), se = 100*sd(correct) / sqrt(n()))
saycam_linear_probes_by_target_category_summary

## fig s2
ggplot(saycam_linear_probes_by_target_category_summary, aes(x = config, y = mean, fill = config)) +
    geom_bar(stat = "identity") +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
    geom_hline(yintercept = 25, linetype = "dashed") +
    # theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_manual(name = "Model", values = c("#33A02C", "#BDD7E7", "#6BAED6", "#1F78B4")) +
    labs(x = "Model", y = "Classification Accuracy") +
    scale_x_discrete(labels = function(x) 
        stringr::str_wrap(x, width = 8)) +
    ylim(0, 100) + 
    theme_bw(base_size=30) +
    theme(legend.position = "bottom",
    #axis.text.x = element_text(angle = 90, hjust = 1)) +
    axis.text.x=element_blank(),
    axis.ticks.x=element_blank()) +
    facet_wrap(~ target_category, labeller = labeller(target_category = capitalize), ncol = 6)

## ggsave("../figures/saycam-linear-probes-by-target-category-summary.pdf", height=14, width=22, units="in", dpi=500) 

## ablation results
saycam_ablations <- read_csv("../results/summary/saycam-ablations.csv")

## extract configs for summary bounds figure
saycam_ablations <- saycam_ablations %>%
    filter(config == "contrastive_embedding" | 
           config == "contrastive_lstm" |
           config == "contrastive_embedding_finetune_random_init" | 
           config == "contrastive_embedding_single_frame")

## rename configs and re-order
saycam_ablations <- saycam_ablations %>%
    mutate(config = ifelse(config == "contrastive_embedding", "CVCL", config)) %>%
    mutate(config = ifelse(config == "contrastive_lstm", "CVCL (LSTM)", config)) %>%
    mutate(config = ifelse(config == "contrastive_embedding_finetune_random_init", "CVCL (Scratch)", config)) %>%
    mutate(config = ifelse(config == "contrastive_embedding_single_frame", "CVCL (Single Frame)", config)) %>%
    mutate(config = factor(config, levels = c("CVCL", "CVCL (LSTM)", "CVCL (Single Frame)", "CVCL (Scratch)")))

## get summary stats for ablations
saycam_ablations_summary <- saycam_ablations %>%
    group_by(config, seed) %>%
    summarise(correct = mean(correct)) %>%
    group_by(config) %>%
    summarise(mean = 100*mean(correct), se = 100*sd(correct) / sqrt(n()))
saycam_ablations_summary

## bar plot of ablations
## fig 2c
ggplot(saycam_ablations_summary, aes(x = config, y = mean, fill=config)) +
    geom_bar(stat = "identity") +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
    geom_text(aes(label=format(round(mean, digits=1), nsmall=1), y=0), position=position_dodge(width=0.9), size = 8, vjust=-0.25) +
    geom_hline(yintercept = 25, linetype = "dashed") +
    # theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_manual(values=c("#33A02C", "#74C476", "#A1D99B", "#C7E9C0")) +
    labs(x = "Model", y = "Classification Accuracy") +
    scale_x_discrete(labels = function(x) 
        stringr::str_wrap(x, width = 8)) +
    ylim(0, 100) + 
    theme_bw(base_size=30) +
    theme(legend.position = "none",
          axis.text.x = element_text(size=20))

## save figure
## ggsave("../figures/saycam-ablations-summary.pdf", height=7, width=7, units="in", dpi=500) 

## object categories
object_categories <- read_csv("../results/summary/object-categories.csv", col_types = cols(split = col_character()))

## get object categories JSON file for filenames
library(rjson)
object_categories_eval <- fromJSON(file="../data/eval_object_categories.json")$data
object_categories_eval_df <- list()
for (i in 1:length(object_categories_eval)) {
    curr_trial <- object_categories_eval[[i]][c("target_img_filename")]
    filename <- curr_trial$target_img_filename
    filename <- str_split(filename, "/")[[1]]
    filename <- paste0(filename[length(filename)-1], "/", filename[length(filename)])
    curr_trial$trial_idx <- i-1
    curr_trial$target_img_filename <- filename
    object_categories_eval_df[[i]] <- curr_trial
}

## add info to data frame
object_categories_eval_df <- bind_rows(object_categories_eval_df)
object_categories <- object_categories %>%
    left_join(object_categories_eval_df, by="trial_idx") 

## correctly filter linear probe trials for cross-validation
object_categories_trials <- object_categories %>%
    group_by(target_category, target_img_filename) %>%
    summarise() %>%
    group_by(target_category) %>%
    mutate(order = ifelse(row_number() <= as.integer(n()/2), "first", "last")) %>%
    ungroup() %>%
    select(target_img_filename, order)

## filter training trials for linear probe    
object_categories <- object_categories %>%
    left_join(object_categories_trials, by="target_img_filename") %>%
    filter(!((config == "linear_probe" & split == "first" & order == "first") |
             (config == "linear_probe" & split == "last" & order == "last")))

## get overall summary across conditions
object_categories_bounds_summary <- object_categories %>%
    group_by(config, seed) %>%
    summarise(correct = mean(correct)) %>%
    group_by(config) %>%
    summarise(mean = 100*mean(correct), se = 100*sd(correct) / sqrt(n())) %>%
    mutate(config = ifelse(config == "contrastive", "CVCL", config)) %>%
    mutate(config = ifelse(config == "contrastive_shuffled", "CVCL (Shuffled)", config)) %>%
    mutate(config = ifelse(config == "contrastive_frozen_random_init", "CVCL (Rand. Features)", config)) %>%
    mutate(config = ifelse(config == "clip", "CLIP", config)) %>%
    mutate(config = ifelse(config == "linear_probe", "Linear Probe", config)) %>%
    mutate(config = factor(config, levels = c("CVCL", "CVCL (Shuffled)", "CVCL (Rand. Features)", "CLIP", "Linear Probe")))

## create bar plot, fig 3a
ggplot(object_categories_bounds_summary, aes(x = config, y = mean, fill=config)) +
    geom_bar(stat = "identity") +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
    geom_hline(yintercept = 25, linetype = "dashed") +
    geom_text(aes(label=format(round(mean, digits=1), nsmall=1), y=0), position=position_dodge(width=0.9), size = 8, vjust=-0.25) +
    # theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    # scale_fill_brewer(palette="Spectral") +
    scale_fill_manual(values = c("#33A02C", "#FED976", "#FDBF6F", "#A6CEE3", "#1F78B4")) +
    labs(x = "Model", y = "Classification Accuracy") +
    scale_x_discrete(labels = function(x) 
        stringr::str_wrap(x, width = 6)) +
    ylim(0, 100) + 
    theme_bw(base_size=30) +
    theme(legend.position = "none",
          axis.text.x = element_text(size=20))

## ggsave("../figures/object-categories-bounds-summary.pdf", height=7, width=7, units="in", dpi=500) 

## summary plot by category
object_categories_bounds_summary_by_category <- object_categories %>%
    group_by(config, seed, target_category) %>%
    summarise(correct = mean(correct)) %>%
    group_by(config, target_category) %>%
    summarise(mean = 100*mean(correct), se = 100*sd(correct) / sqrt(n())) %>%
    mutate(config = ifelse(config == "contrastive", "CVCL", config)) %>%
    mutate(config = ifelse(config == "contrastive_shuffled", "CVCL (Shuffled)", config)) %>%
    mutate(config = ifelse(config == "contrastive_frozen_random_init", "CVCL (Rand. Features)", config)) %>%
    mutate(config = ifelse(config == "clip", "CLIP", config)) %>%
    mutate(config = ifelse(config == "linear_probe", "Linear Probe", config)) %>%
    mutate(config = factor(config, levels = c("CVCL", "CVCL (Shuffled)", "CVCL (Rand. Features)", "CLIP", "Linear Probe")))

ggplot(object_categories_bounds_summary_by_category, aes(x = config, y = mean, fill = config)) +
    geom_bar(stat = "identity") +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
    geom_hline(yintercept = 25, linetype = "dashed") +
    scale_fill_manual(values = c("#33A02C", "#FED976", "#FDBF6F", "#A6CEE3", "#1F78B4")) +
    labs(x = "Model", y = "Classification Accuracy") +
    scale_x_discrete(labels = function(x) 
        stringr::str_wrap(x, width = 10)) +
    ylim(0, 100) + 
    theme_bw(base_size=16) +
    theme(legend.position = "bottom",
          #axis.text.x = element_text(angle = 90, hjust = 1)) +
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank()) +
    facet_wrap(~ target_category, labeller = labeller(target_category = capitalize), ncol = 8)

## compare cvc and linear probe
object_categories_cvc_vs_linear_probe <- object_categories_bounds_summary_by_category %>%
    filter(config == "Linear Probe" | config == "CVCL") %>%
    select(-se) %>%
    spread(config, mean)

object_categories_rank_cor <- cor.test(
    object_categories_cvc_vs_linear_probe$CVCL,
    object_categories_cvc_vs_linear_probe$`Linear Probe`,
    method="spearman")
print(object_categories_rank_cor)

## get frequency info
object_categories_freq <- c(ball = 481, train = 235, socks = 129, bottle = 110, camera = 92, pants = 91, 
          apple = 77, watch = 73, balloon = 68, bucket = 59, chair = 54, spoon = 53, 
          crib = 51, jacket = 49, juice = 48, bowl = 46, tree = 46, backpack = 44, 
          bed = 43, bird = 36, button = 34, shoe = 34, dog = 31, hat = 28, pen = 27, 
          leaves = 26, cat = 25, bike = 23, butterfly = 23, cake = 22, guitar = 21, 
          basket = 19, umbrella = 18, phone = 17, knife = 16, bagel = 11, bench = 10, 
          cheese = 10, clock = 10, key = 10, hairbrush = 9, rock = 9, turtle = 9, 
          airplane = 8, ring = 7, sofa = 7, broom = 6, stool = 6, bell = 5, cookie = 5, 
          microwave = 5, scissors = 5, stamp = 5, tv = 5, coin = 4, necklace = 4, 
          sandwich = 4, toothpaste = 4, desk = 3, fan = 3, kayak = 3, pipe = 3, 
          pizza = 3, tricycle = 3)

# convert the vector to a data frame
object_categories_freq <- tibble(target_category = names(object_categories_freq), freq = object_categories_freq)
object_categories_freq

## filter cvc results only
object_categories_cvc <- object_categories %>%
    filter(config == "contrastive")

## get individual category performance
curr_category <- c("butterfly", "bucket", "button", "spoon")
for(category in curr_category) {
    curr_category_summary <- object_categories_cvc %>% 
        filter(target_category == category) %>% 
        group_by(target_img_filename) %>% 
        summarise(correct = 100*mean(correct)) %>% 
        arrange(desc(correct))
    print(curr_category_summary[c(1, 2, 7, 15), ])
}

## get summary
object_categories_cvc_summary <- object_categories_cvc %>%
    group_by(target_category, seed) %>%
    summarise(correct = mean(correct)) %>%
    group_by(target_category) %>%
    summarise(mean = 100*mean(correct), se = 100*sd(correct) / sqrt(n())) %>%
    arrange(-mean) %>%
    mutate(full = c(rep("Top", 32), rep("Bottom", 32))) %>%
    mutate(full = factor(full, levels=c("Top", "Bottom"))) %>%
    left_join(object_categories_freq, by="target_category") %>%
    mutate(labels = paste0(target_category, " (", freq, ")")) %>%
    mutate(target_category = (fct_reorder(target_category, mean, .desc=TRUE)))

## plot of all categories
ggplot() +
    geom_point(data=object_categories_cvc_summary, size=2.5,
               aes(x = target_category, y = mean)) +
    geom_errorbar(data=object_categories_cvc_summary, aes(x = target_category, y = mean, ymin = mean - se, ymax = mean + se), width = 0.4) +
    geom_hline(yintercept = 25, linetype = "dashed") +
    geom_hline(data=object_categories_bounds_summary, linetype="dashed",
               size=1,
               aes(yintercept=mean, color=config)) +
    geom_text_repel(data=object_categories_bounds_summary, size=6,
                    force=0.5, hjust=0, xlim = c(0, 50), ylim = c(0, 120), direction="y",
                    nudge_x=0, nudge_y=2, min.segment.length = Inf,
                    aes(x=33, y=mean-3, label=config, color=config)) +
    scale_color_manual(values = c("#33A02C", "#FED976", "#FDBF6F", "#A6CEE3", "#1F78B4")) +
    coord_cartesian(clip = "off") +
    facet_wrap(~ full, scale='free_x', nrow=2) +
    labs(x = "Target Category", y = "Classification Accuracy") +
    ylim(0, 100) + 
    theme_bw(base_size=22) +
    theme(legend.position = "none",
          plot.margin = margin(0.5,6.5,0.5,0.5, "cm"),
          axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
          strip.text.x = element_blank())
          # panel.spacing = unit(2, "lines"))    

## ggsave("../figures/object-categories-by-target-category-summary-full.pdf", 
##        height=8, width=14, units="in", dpi=500)

## scatterplot with accuracy and frequency
ggplot(object_categories_cvc_summary, aes(x = log(freq), y = mean, label=labels)) +
    geom_point() +
    geom_smooth(method = 'lm') +
    geom_label_repel(max.overlaps = Inf) +
    theme_bw(base_size=24)

## embedding visualization
## fig. 4a
embeddings <- read_csv("../results/alignment/joint_embeddings_with_eval_sims_seed_0.csv")

## group categories
embeddings <- embeddings %>%
    rename(cat = kitty) %>%
    mutate(eval_category = ifelse(eval_category == "kitty", "cat", eval_category)) %>%
    mutate(group = case_when(eval_category %in% c("sand", "crib", "car", "paper") ~ 1,
                             eval_category %in% c("stairs", "puzzle", "ball", "road") ~ 2,
                             eval_category %in% c("chair", "cat", "computer", "window") ~ 3,
                             eval_category %in% c("kitchen", "ground", "table", "door") ~ 4,
                             eval_category %in% c("foot", "floor", "toy", "basket") ~ 5,
                             eval_category %in% c("room", "hand") ~ 6,
                             TRUE ~ 0))

## get mean image and text embeddings
mean_image_text_embeddings <- embeddings %>%
    filter(embedding_type == "image_mean" | embedding_type == "text")

## only want labels for one set
labels <- mean_image_text_embeddings %>%
    pull(eval_category)
for (i in 1:(length(labels)/2)) {
    labels[i] <- ""
}
text_labels <- labels[23:44]

## mean embedding plot, fig 4b
mean_image_text_embeddings <- mean_image_text_embeddings %>%
    mutate(embedding_type = case_when(
        embedding_type == "image_mean" ~ "Image",
        embedding_type == "text" ~ "Text"
    ))

ggplot(mean_image_text_embeddings, aes(x = x, y = y, color = embedding_type, group = eval_category)) +
    geom_line(linetype = 2, color = "grey30", size=1) +
    geom_text_repel(label = labels, color = "black", min.segment.length = 0, segment.color = "grey50",
                    nudge_x = -5, nudge_y = -5, size = 10, force = 3, box.padding=1.5, max.overlaps = Inf) +
    geom_point(size = 5, alpha=0.8) +
    scale_color_manual(name = "Embedding", values = c(brewer.pal(9, "Blues")[7], brewer.pal(9, "Greens")[4])) +
    #scale_color_manual(name = "Embedding", values = c("white", brewer.pal(9, "Greens")[4])) +
    theme_bw(base_size=30) +
    xlab("") +
    ylab("") +
    theme(
       legend.position = "bottom",
       legend.title = element_text(size = 24),
       legend.margin = margin(-40, 0, 0, 0),
       axis.text.x = element_blank(),
       axis.text.y = element_blank(),
       axis.ticks.x = element_blank(),
       axis.ticks.y = element_blank())

## ggsave("../figures/joint-tsne-text-only.pdf", height=10, width=10, units="in", dpi=500)

## calculate correlation between visual prototype and text embedding
mean_image_text_embeddings_dist <- mean_image_text_embeddings %>%
    group_by(eval_category) %>%
    summarise(dist = sqrt(sum((c(first(x), first(y)) - c(last(x), last(y)))^2))) %>%
    mutate(correct = saycam_bounds_by_target_category_summary %>% filter(config == "CVCL") %>% pull(mean))

## calculate correlation
cor.test(mean_image_text_embeddings_dist$dist,
         mean_image_text_embeddings_dist$correct)

## scatterplot to show correlation, fig s5
ggplot(mean_image_text_embeddings_dist,
       aes(x = dist, y = correct, label=eval_category)) +
    geom_point(size = 3, aes(color = eval_category)) +
    geom_smooth(method = 'lm') +
    geom_label_repel(size=5, force=0.5, min.segment.length = 0, box.padding=1, 
                     nudge_x = 8, nudge_y = -4, 
                     aes(color = eval_category)) +
    labs(x = "Euclidean Distance in t-SNE space", y = "Classification Accuracy") +
    theme_bw(base_size=24) +
    theme(legend.position = "none")

## ggsave("../figures/embedding-distance-vs-classification-performance-scatterplot.pdf", height=10, width=10, units="in", dpi=500)

## calculate correlation between mean distance to prototype and performance
embeddings_only <- embeddings %>%
    filter(embedding_type == "image")

## rename mean_image_text_embeddings
mean_image_text_embeddings_only <- mean_image_text_embeddings %>%
    filter(embedding_type == "Image") %>%
    rename(mean_x = x, mean_y = y) %>%
    select(eval_category, mean_x, mean_y)

## perform join between embeddigs
embeddings_only <- embeddings_only %>%
    left_join(mean_image_text_embeddings_only, by="eval_category") %>%
    select(eval_category, x, y, mean_x, mean_y)

## compute average distance to centroid
embeddings_avg_dist <- embeddings_only %>%
    mutate(dist = sqrt((x - mean_x)^2 + (y - mean_y)^2)) %>%
    group_by(eval_category) %>%
    summarise(dist = mean(dist)) %>%
    mutate(correct = saycam_bounds_by_target_category_summary %>% filter(config == "CVCL") %>% pull(mean))

## compute correlation
cor.test(embeddings_avg_dist$dist,
         embeddings_avg_dist$correct)

## set factor of embeddings based on classification performance
embeddings$eval_category <- factor(embeddings$eval_category, target_category_factor_ordering)

## plot embeddings across all categories
## Define a vector of 22 different colors
colors <- sample(c("#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
            "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
            "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
            "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#FFD700", "#E41A1C"), 22, replace=FALSE)

## note to self: used the following hack to set the text to black, but keep the label border varying to match the scatterplot points of each category https://stackoverflow.com/questions/67058977/border-of-different-color-in-geom-label-repel
ggplot(embeddings %>% filter(embedding_type == "image") %>% select(-group), aes(x = x, y = y)) +
    geom_point(color="grey90") +
    geom_line(data = embeddings %>% filter(embedding_type == "image_mean" | embedding_type == "text"), aes(group = eval_category),
                  linetype = 2, color = "grey30", size=0.5) +
    geom_point(data=embeddings %>% filter(embedding_type == "image"),
       aes(x = x, y = y, color = eval_category), alpha=0.7) +
    geom_point(data=embeddings %>% filter(embedding_type == "text"), 
               aes(x = x, y = y, fill=eval_category), color="black", shape=23, stroke=1, size = 3) +
    geom_point(data=embeddings %>% filter(embedding_type == "image_mean"), 
               aes(x = x, y = y, fill=eval_category), color="black", shape=21, stroke=1, size=3) +
    geom_label_repel(data=embeddings %>% filter(embedding_type == "text"), 
                     aes(x = x, y = y, label = eval_category, size=8,
                         color = eval_category)) +    
    theme_bw(base_size=30) +
    coord_equal() +
    facet_wrap(~ group) +
    xlab("") +
    ylab("") +
    scale_color_manual(values = colors) +
    scale_fill_manual(values = colors) + 
    theme(legend.position = "none",
          strip.text.x = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.x = element_blank(),
          axis.ticks.y = element_blank())

## ggsave("../figures/joint-tsne-all.pdf", height=7, width=10, units="in", dpi=500)

## per category embeddings
## fig s6

set.seed(42)
all_plots <- list()
all_categories <- embeddings %>% pull(eval_category) %>% unique()
idx <- 1
for (curr_category in all_categories) {
    embeddings <- embeddings %>%
        mutate(is_x = ifelse(eval_category == curr_category, TRUE, FALSE)) 
    embeddings_sim_filtered <- embeddings %>%
        top_n(100, eval(parse(text=curr_category)))
    
    a <- ggplot(embeddings %>% filter(!is_x), aes(x=x, y=y)) +
        geom_point(alpha=0.1, size=0.75, color="grey70") +
        geom_point(aes(color=is_x, group=is_x), size=0.75, color="#2B83BA", alpha = 0.8, data = embeddings %>% filter(is_x)) +
        coord_fixed() +
        theme_bw(base_size=18) +
        xlab("") +
        ylab("") +
        ggtitle(str_wrap(glue("Label:\n{capitalize(curr_category)}"), 8)) +
        theme(
            plot.title = element_text(size = 10),
            plot.margin = margin(-100, -50, -100, -50, "pt"),
            legend.position = "none",
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            axis.ticks.x = element_blank(),
            axis.ticks.y = element_blank())
    
    b <- ggplot(embeddings, aes(x = x, y = y)) +
        geom_point(alpha=0.1, size=0.75, color="grey70") +
        geom_point(data=embeddings_sim_filtered, size=0.75, alpha=0.8, color="#1A9850") +
        coord_fixed() +
        #geom_text_repel(data=embeddings_filtered, aes(x=tsne_1, y=tsne_2, label=filename), max.overlaps = Inf) +
        scale_fill_brewer() +
        theme_bw(base_size=12) +
        xlab("") +
        ylab("") +
        ggtitle(str_wrap(glue("Most similar:\n{capitalize(curr_category)}"), 13)) +
        theme(
            plot.title = element_text(size = 10),
            plot.margin = margin(-100, -50, -100, -50, "pt"),
            legend.position = "none",
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            axis.ticks.x = element_blank(),
            axis.ticks.y = element_blank())
    
    all_plots[[idx]] <- a
    all_plots[[idx+1]] <- b
    idx <- idx + 2
}    
wrap_plots(all_plots, ncol=8) + plot_layout(widths=c(), heights=c())

## ggsave("../figures/tsne-label-most-sim-all.pdf", height=12, width=12, units="in", dpi=500) 

## remake plot for three categories (sand, puzzle, stairs)
## fig 4c
subset_categories <- c("sand", "stairs", "puzzle")
for (curr_category in subset_categories) {
    embeddings <- embeddings %>%
        mutate(is_x = ifelse(eval_category == curr_category, TRUE, FALSE)) 
    embeddings_sim_filtered <- embeddings %>%
        top_n(100, eval(parse(text=curr_category)))
    
    a <- ggplot(embeddings %>% filter(!is_x), aes(x=x, y=y)) +
        geom_point(alpha=0.1, size=1.5, color="grey70") +
        geom_point(aes(color=is_x, group=is_x), size=1.5, color="#2B83BA", alpha = 0.8, data = embeddings %>% filter(is_x)) +
        geom_text_repel(data=embeddings %>% filter(is_x) %>% sample_n(20), aes(label=image_filename), max.overlaps=Inf, box.padding = 0.2, size=4) +
        coord_fixed() +
        theme_bw(base_size=20) +
        xlab("") +
        ylab("") +
        ggtitle(str_wrap(glue("Label:\n{capitalize(curr_category)}"), 8)) +
        theme(
            plot.title = element_text(size = 10),
            plot.margin = margin(0, 0, 0, 0, "pt"),
            legend.position = "none",
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            axis.ticks.x = element_blank(),
            axis.ticks.y = element_blank())
    
    b <- ggplot(embeddings, aes(x = x, y = y)) +
        geom_point(alpha=0.1, size=1.5, color="grey70") +
        geom_point(data=embeddings_sim_filtered, size=1.5, alpha=0.8, color="#1A9850") +
        coord_fixed() +
        geom_text_repel(data=embeddings_sim_filtered %>% sample_n(20), aes(label=image_filename), max.overlaps=Inf, box.padding = 0.2, size=4) +
        scale_fill_brewer() +
        theme_bw(base_size=20) +
        xlab("") +
        ylab("") +
        ggtitle(str_wrap(glue("Most similar:\n{capitalize(curr_category)}"), 13)) +
        theme(
            plot.title = element_text(size = 10),
            plot.margin = margin(0, 0, 0, 0, "pt"),
            legend.position = "none",
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            axis.ticks.x = element_blank(),
            axis.ticks.y = element_blank())
    (a|b)
    ## ggsave(glue("../figures/tsne-label-most-sim-{curr_category}-with-filenames.pdf"), height=8, width=12, units="in", dpi=500) 
}

## overlap plot
matched_df <- read_csv("../results/duplicates/matched_results.csv")

## convert cosine sims into bins of width 0.05
matched_df <- matched_df %>%
    mutate(bin = ((cosine_sim * 20) %/% 1) / 20)

## fig s8
ggplot(matched_df, aes(x = bin, fill = matched)) +
    geom_bar(stat="count", position=position_dodge2(preserve="single"), color="black", width=0.04) +
    xlab("Cosine Similarity of Nearest Training Frame") +
    ## scale_x_continuous(breaks=seq(0.5, 1.0, by=0.1)) +
    xlim(0.5, 1) +
    scale_fill_manual(breaks = c("match", "mismatch"), values = c("match" = "#33A02C", "mismatch" = "#88D78F"), labels = c("Match", "Mismatch")) +
    labs(fill = "Matched evaluation category in utterance") +
    ylab("Count") +
    theme_bw(base_size=18) +
    theme(legend.position = "bottom")

## ggsave("../figures/cosine-similarity-indirect-overlap.pdf", height=7, width=8)
