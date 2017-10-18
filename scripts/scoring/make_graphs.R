library(ggplot2)

data <- read.csv("concat.csv")
data$class <- as.factor(data$class)

ggplot(data, aes(x=probability, fill=class)) +
  geom_histogram(binwidth=0.05) +
  facet_grid(class ~ ., scales="free_y") +
  scale_y_log10()

