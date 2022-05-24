# Survival analysis script
library(tidyverse)
library(survival)
library(survminer)

#setwd("~/Documents/Médecine/Recherche/CODA19/git/v7_submission/R Scripts")

# Loading and preparing the data
# Dataset containing all the info to perform survival analysis on death, MV and ICU admission
clusters <- read.csv("./data/other/clusters_pre_survival.csv")%>%select(-X)

# Level of care/POLST data
# Contains cases for which LOC< 2
coda_loc <- read_csv("./data/other/loc_inclus.csv")%>%drop_na()
#clusters_loc <- clusters%>%right_join(coda_loc, by = "patient_site_uid")
clusters <- coda_loc%>%left_join(clusters, by = "patient_site_uid")



# formatting the dataframe for survivla function (1 col with time, 1 col with 0/1)
clusters <- clusters%>%rename("cluster"="clusters")
clusters <- clusters%>%mutate(
  death = if_else(
    is.na(death_event), 0, 1),
  mv = if_else(
    is.na(mv_event), 0, 1),
  icu = if_else(
    is.na(icu_event), 0, 1),
  )

# Creating formal time-to-survival column (duration time or event time - whichever is first)
clusters <- clusters%>%mutate(
  death_survival=
    case_when(
      death == 1 ~ death_event,
      death == 0 ~ as.integer(obs_time)
    )
  )

clusters <- clusters%>%mutate(
  icu_survival=
    case_when(
      icu == 1 ~ icu_event,
      icu == 0 ~ as.integer(obs_time)
    )
)


clusters <- clusters%>%mutate(
  mv_survival=
    case_when(
      mv == 1 ~ mv_event,
      mv == 0 ~ as.integer(obs_time)
    )
)



# Survival Analysis -------------------------------------------------------



###### DEATH risk

# Log rank test
survdiff(Surv(clusters$death_survival, clusters$death) ~ cluster, data = clusters)

# Survival function for death as outcome
surv_compare = survfit(Surv(clusters$death_survival, clusters$death) ~ cluster, data = clusters)
# Calculating 30 day mortality risk
summary(surv_compare, times = 30)

# # Cum death risk data
# surv_compare_death = survfit(Surv(clusters$death_survival, clusters$discharge) ~ cluster, data = clusters)
# # Comparing p-value at time x https://stats.stackexchange.com/questions/363244/r-survfit-function-getting-p-value-for-a-specified-time-period
# summary(surv_compare_death, times = 30)


# Cmulative death plot
j <- ggsurvplot(
  surv_compare,
  data = clusters,
  xlim = c(0,50),
  xlab = "Days", 
  ylab = "Cumulative death",
  fun = "event",
  pval = FALSE, 
  pval.method = TRUE,
  pval.size = 3,
  pval.coord = c(45,1.00),
  pval.method.coord = c(45,1.03),
  palette = c("#4b595d", "#dc8e47", "#05a4d4"),
  #risk.table = TRUE,
  linetype = "strata",
  #conf.int = TRUE,
  ggtheme = theme_minimal(),
  legend = "bottom",
  break.time.by = 5
)+ 
  theme_survminer(
    font.main = c(16, "bold", "darkblue"),
    font.submain = c(15, "bold.italic", "purple"),
    font.caption = c(14, "plain", "orange"),
    font.x = c(14, "plain"),
    font.y = c(14, "plain"),
    font.tickslab = c(12, "plain"),
    font.legend = c(size = 14, color = "black", face = "bold"),
    legend = "bottom"
  ) 

j$plot <- j$plot +
  annotate("text", x = 47, y = 1.03, label = "Log-Rank", fontface=2) +
  annotate("text", x = 44.5, y = 1.00, label = "p", fontface=4) +
  annotate("text", x = 47, y = 1.00, label = "=0.012", fontface=2)
j

######  ICU admission risk
# Log Rank
surv_compare_icu = survfit(Surv(clusters$icu_survival, clusters$icu) ~ cluster, data = clusters)
survdiff(Surv(clusters$icu_survival, clusters$icu) ~ cluster, data = clusters)


# 1) Cumulative icu
j <- ggsurvplot(
  surv_compare_icu,
  data = clusters,
  palette = c("#4b595d", "#dc8e47", "#05a4d4"),
 #pval = TRUE, 
  pval.method = TRUE,
  pval.size = 4,
  pval.coord = c(9,0.5),
  pval.method.coord = c(7,0.5),
  #conf.int = TRUE,
  #conf.int.style = "ribbon",
  surv.median.line = "hv",
  ylim = c(0,1.05),
  xlim = c(0,11),
  xlab = "Days", 
  ylab = "Cumulative ICU admission",
  fun = "event",
  #risk.table = TRUE,
  linetype = "strata",
  #conf.int = TRUE, 
  legend = "bottom",
  ggtheme = theme_minimal(),
  break.time.by = 5
)+ 
  theme_survminer(
    font.main = c(16, "bold", "darkblue"),
    font.submain = c(15, "bold.italic", "purple"),
    font.caption = c(14, "plain", "orange"),
    font.x = c(14, "plain"),
    font.y = c(14, "plain"),
    font.tickslab = c(12, "plain"),
    font.legend = c(size = 14, color = "black", face = "bold"),
    legend = "bottom"
  ) 

j$plot <- j$plot +
  annotate("text", x = 10.5, y = 1.03, label = "Log-Rank", fontface=2) +
  annotate("text", x = 10, y = 1.00, label = "p", fontface=4) +
  annotate("text", x = 10.7, y = 1.00, label = "< 0.0001", fontface=2)
j

  
summary(surv_compare_icu)

# Calculating 7 day ICU admission risk
summary(surv_compare_icu, times = 7)



######  Mechanival Ventilation risk

# Log Rank
surv_compare_mv = survfit(Surv(clusters$mv_survival, clusters$mv) ~ cluster, data = clusters)
survdiff(Surv(clusters$mv_survival, clusters$mv) ~ cluster, data = clusters)


# 1) Cumulative risk of MV
j <- ggsurvplot(
  surv_compare_mv,
  data = clusters,
  palette = c("#4b595d", "#dc8e47", "#05a4d4"),
  xlim = c(0,25),
  ylim = c(0,1.05),
  xlab = "Days", 
  ylab = "Cumulative Mechanical Ventilation",
  fun = "event",
  #pval = TRUE, 
  pval.method = TRUE,
  pval.size = 3,
  pval.coord = c(20,0.75),
  pval.method.coord = c(20,0.78),
  #risk.table = TRUE,
  linetype = "strata",
  #conf.int = TRUE, 
  legend = "bottom",
  ggtheme = theme_minimal(),
  break.time.by = 5
)+ 
  theme_survminer(
    font.main = c(16, "bold", "darkblue"),
    font.submain = c(15, "bold.italic", "purple"),
    font.caption = c(14, "plain", "orange"),
    font.x = c(14, "plain"),
    font.y = c(14, "plain"),
    font.tickslab = c(12, "plain"),
    font.legend = c(size = 14, color = "black", face = "bold"),
    legend = "bottom"
  ) 

j$plot <- j$plot +
  annotate("text", x = 21.1, y = 1.03, label = "Log-Rank", fontface=2) +
  annotate("text", x = 20, y = 1.00, label = "p", fontface=4) +
  annotate("text", x = 21.5, y = 1.00, label = "< 0.0001", fontface=2)
j


# Calculating 7 day risk of MV
summary(surv_compare_mv, times = 7)
# # Comparing cluster 2 vs 3 only
# subset <- clusters%>%filter(cluster%in%c(2,3))
# survdiff(Surv(subset$mv_survival, subset$mv) ~ cluster, data = subset)



