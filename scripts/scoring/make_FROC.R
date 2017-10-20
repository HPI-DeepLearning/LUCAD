library(ggplot2)
library(tidyr)

comparison <- "name,0.125,0.25,0.5,1,2,4,8,fp_average
CUMedVis,0.677,0.834,0.927,0.972,0.981,0.983,0.983,0.908
JackFPR,0.734,0.796,0.859,0.892,0.923,0.944,0.954,0.872
DIAG CONVNET,0.669,0.760,0.831,0.892,0.923,0.945,0.960,0.854
CADIMI,0.583,0.677,0.743,0.815,0.857,0.893,0.916,0.783
ZNET,0.511,0.630,0.720,0.793,0.850,0.884,0.915,0.758"

data.comparison <- read.csv(text=comparison)
data.comparison <- gather(data.comparison, key="fp_rate", value="Sensivity_Mean_", X0.125, X0.25, X0.5, X1, X2, X4, X8)
data.comparison$fp_rate <- as.factor(data.comparison$fp_rate)
data.comparison$fp_rate <- gsub('X', '', data.comparison$fp_rate)
data.comparison$Sensivity_Lower_bound_ <- NA
data.comparison$Sensivity_Upper_bound_ <- NA

data.own <- read.csv("results.csv")
data.own$fp_rate <- as.factor(data.own$fp_rate)

# only show best model, comment out to disable
#data.own <- data.own[data.own$fp_average == max(data.own$fp_average),]

# only show best model, comment out to disable
#data.own <- data.own[data.own$fp_average >= 0.83,]

# levels(data.own$name)[match("probability_avg_stage_A_0.50", levels(data.own$name))] <- "Our Model"

data <- rbind(data.own, data.comparison)

p1 <- ggplot(data, aes(x=fp_rate, y=Sensivity_Mean_, colour=name, group=name)) +
  geom_line() +
  geom_line(aes(y=data$Sensivity_Upper_bound), linetype=3) +
  geom_line(aes(y=data$Sensivity_Lower_bound), linetype=3) +
  ylim(c(0,1))
ggsave("rate.png", plot = p1, width = 9, height = 4)

data.average <- data[data$fp_rate == '1',]
data.average <- transform(data.average, name=reorder(name, -fp_average) ) 
p2 <- ggplot(data.average, aes(x=name, y=fp_average, fill=name)) +
  geom_bar(stat="identity", position="dodge") +
  ylim(c(0,1))
ggsave("average.png", plot = p2, width = 9, height = 4)
