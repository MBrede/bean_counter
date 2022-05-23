library(jsonlite)
library(ggplot2)
library(purrr)
library(dplyr)

p = map_dfr(list.files(pattern='json'), ~ fromJSON(.)) %>%
    mutate(ks = na_if(ks, 99)) %>%
    ggplot(aes(x = image, y = ks)) +
    geom_boxplot() +
    ylim(0,1) +
    coord_flip() 
ggsave(plot = p,'current_results.png')
