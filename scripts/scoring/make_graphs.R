library(ggplot2)

data <- read.csv("concat.csv")
data$class <- as.factor(data$class)

p <- ggplot(data, aes(x=probability, fill=class)) +
  stat_bin(binwidth=0.05, position="dodge") +
  stat_bin(binwidth=0.05, geom="text", aes(label=..count..), vjust=-0.5, size=3, position="dodge") +
  coord_cartesian(ylim=c(0, 1500))
ggsave("histogram_fixed.png", plot = p, width = 9, height = 4)

p <- ggplot(data, aes(x=probability, fill=class)) +
  stat_bin(binwidth=0.05, position="dodge") +
  stat_bin(binwidth=0.05, geom="text", aes(label=..count..), vjust=+1.5, size=3, position="dodge") +
  scale_y_log10() +
  facet_grid(class ~ ., scales="free_y")
ggsave("histogram_log.png", plot = p, width = 9, height = 4)
