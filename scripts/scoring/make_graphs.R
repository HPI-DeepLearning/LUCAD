library(ggplot2)

data <- read.csv("concat.csv")
data$class <- as.factor(data$class)

ggplot(data, aes(x=probability, fill=class)) +
  stat_bin(binwidth=0.05, position="dodge") +
  stat_bin(binwidth=0.05, geom="text", aes(label=..count..), vjust=-0.5, size=3, position="dodge") + 
  coord_cartesian(ylim=c(0, 1500))
