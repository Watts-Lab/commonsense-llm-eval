library(lmerTest)

data <- read.csv("data/mixed_effect_size_vs_commonsensicality.csv")

result <- lmer(commonsensicality ~ log_size + (1 | model_family), data=data)
summary(result)
confint(result)