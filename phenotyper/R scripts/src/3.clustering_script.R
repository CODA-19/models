# Library 

source("utils_template.R")
# Loading workspace
#load("CODA19_phenotyper_new.RData")

## specify the packages needed

pkgs <- c("bookdown","DT", "leaflet", "splines2", "webshot","autoEDA", "bigrquery","plotly","scales","RColorBrewer","data.table","tidyverse","knitr","corrplot",
          "cluster", "Rtsne", "FactoMineR", "factoextra", 'fpc', 'NbClust', 'rjson',
          "Hmisc","stats", "janitor", "lubridate", "testthat", "magrittr", "bookdown",
          "purrr", "healthcareai", "RSQLite", "comorbidity", "DataExplorer", "skimr", "summarytools",
          "explore", "dataMaid", "explor", "slickR", "htmlwidgets", "kableExtra",
          "cluster", "Rtsne", "cowplot", "svglite", "FactoMineR", "factoextra", "fpc", "NbClust",
          "FactoMineR", "factoextra", 'optCluster','BiocManager', 'scmap', 'randomForest', 'rpart', 'party', 'ggparty',
          'rattle', 'alookr', 'dlookr')

need.packages(pkgs)
#devtools::install_github("davidsjoberg/ggsankey")
library(ggsankey)
library(FactoMineR)
library(dplyr)
library(readr)

# Separating categorical from numerical variables
transform_data <- function(coda19) {
  coda19 <- coda19%>%mutate_if(function(x){(length(unique(x)) < 9)}, function(x){as.factor(x)})
  coda19 <- coda19%>%mutate_if(function(x){(length(unique(x)) >= 9)}, function(x){as.numeric(x)})
  return (coda19)
}

# Function allowing the selection of final set of candidate variables
variables_selection_complete <- function(df){
  selectedVar <- c("patient_site_uid",
                   # demographics
                   "patient_age","male",
                   # hematologic
                   "hemoglobin", "plt", "wbc", "neutrophil","lymph", "vgm", "vpm","nlr",
                   # renal
                   "sodium", "potassium", "creatinine", "bun", "bicarbonate", 
                   # hemodynamics
                   "sbp", "dbp", "rr", "hr", "shock_index", "anion_gap", "tropot", "bnp",
                   # inflammatory                   
                   "albumin", "d_dimer", "crp", "ldh", "lactate", "ck","ferritin","procalcitonin",
                   # gastro
                   "alt", "bili_tot", 
                   # resp
                   "pao2", "fio2", "so2", "temp", "s_f",
                   # imaging
                   "opacities_number","opacities_size",
                   # comorbidities
                   "factor_xa_inhibitors", "anticholesteremic_agents", "bronchodilator_agents",
                   "antihypertensive_agents", "diuretics", "hypoglycemic_agents", "platelet_aggregation_inhibitors",
                   "mci",
                   # outcomes
                   'mv', "wave", "icu", "death")
  
  # check names in df
  df_names <- colnames(df)

  # create NLR
  # determine appropriate columns names (removing missing)
  realVar <- selectedVar[selectedVar%in%df_names]
  df <- df%>%select(realVar)
  return(df)
}

# Function allowing the selection of final set of candidate variables

variables_selection_selected<- function(df){
  selectedVar <- c("patient_site_uid",
                   # demographics
                   "patient_age","male",
                   # hematologic
                   "hemoglobin", "plt", "wbc","vpm","nlr",
                   # renal
                   "sodium", "potassium", "creatinine", "bun",
                   # hemodynamics
                   "dbp", "rr", "shock_index", "anion_gap", "tropot", "bnp",
                   # inflammatory                   
                   "albumin", "d_dimer", "crp", "ldh", "lactate", "ck","ferritin","procalcitonin",
                   # gastro
                   "alt", "bili_tot", 
                   # resp
                   "pao2", "temp", "s_f",
                   # imaging
                   "opacities_number", "opacities_size",
                   # comorbidities
                   "factor_xa_inhibitors", "anticholesteremic_agents", "bronchodilator_agents",
                   "antihypertensive_agents", "diuretics", "hypoglycemic_agents", "platelet_aggregation_inhibitors",
                   # outcomes
                   'mv', "wave", "icu", "death")
  
  # check names in df
  df_names <- colnames(df)
  
  # create NLR
  # determine appropriate columns names (removing missing)
  realVar <- selectedVar[selectedVar%in%df_names]
  df <- df%>%select(realVar)
  return(df)
}


# Dataset Preparation -----------------------------------------------------

# Complete imputed dataset
coda19_original <- read_csv("./data/imputed/covid24h_imputed_original.csv")
# Patients weights data
coda19_weights <- read_csv("./data/other/covid_weights.csv")%>%select(patient_site_uid,weight)
# Length of stay data
coda_los <- read_csv("./data/other/covid_stay_2021.csv")%>%
  mutate(
    los = as.numeric(episode_end_time-episode_start_time)/1440
  )
# Level of care data
coda_loc <- read_csv("./data/other/loc_inclus.csv")

# coda19_consortium<- read_csv("./datasets/covid24h_imputed_categ.csv") # vitals here are not binned
# recalculating nlr with imputed data
coda19_original <- coda19_original%>%mutate(
  "nlr" = neutrophil/lymph)
# replacing inf value
coda19_original$nlr[sapply(coda19_original$nlr, is.infinite)] <- 100

# Computing the MCI score
coda19_original <- coda19_original%>%mutate(
  "mci" = hypoglycemic_agents*2 +
    bronchodilator_agents*2 +
    anticholesteremic_agents +
    antihypertensive_agents +
    platelet_aggregation_inhibitors +
    diuretics +
    factor_xa_inhibitors)

    
# coda19_original <- coda19_original%>%left_join(coda19_weights, by = "patient_site_uid")

coda19_imaging <- read_csv("./data/imaging/new_mapped_surfaces.csv")
coda19_imaging <- coda19_imaging%>%distinct(patient_site_uid, .keep_all=TRUE)%>%select(-c("X1"))
coda19_imaging <- coda19_imaging[,c(3,1,2)]

coda19_original <- transform_data(coda19_original)
coda19_imaging <- transform_data(coda19_imaging)

#### Original
# Will be devided in Imaging and No Imaging

coda19_original_rx0 <- coda19_imaging%>%left_join(coda19_original, by = "patient_site_uid")%>%select(-c("opacities_number", "opacities_size"))
coda19_original_rx <- coda19_imaging%>%left_join(coda19_original, by = "patient_site_uid")

# Variables selection
coda19_original_rx0 <- variables_selection_complete(coda19_original_rx0)
coda19_original_rx <- variables_selection_complete(coda19_original_rx)


# ## Distinguish the wave for the 
# coda19_consortium_wave_first <- coda19_consortium%>%filter(wave==0)
# coda19_consortium_wave_second <- coda19_consortium%>%filter(wave==1)

# 3 consortium dataset 1) missing data included 2) missing 1st wave and 2nd wave 3) excluding the missing dataset





# FAMD -------------------------------------------------------------------------


###### Data prep before running FAMD


# Multiple packages : cor, findCorrelation, c.f. https://stackoverflow.com/questions/18275639/remove-highly-correlated-variables
## 1) Removal of highly correlated variables
alookr::treatment_corr(coda19_original_rx, treat = FALSE)

# cor(coda19_original_rx$neutrophils, coda19_original_rx$wbc)
## 2) Log transformation 
toskew <- find_skewness(coda19_original_rx%>%select(-c('patient_site_uid', 'opacities_size')), index = FALSE, value = TRUE, thres = 0.5)
print(toskew)

# Storing variables to logtransform
toskew_names <- names(toskew)
toskew_names_rx0 <- toskew_names[-14]

# coda19_skewed_rx <- coda19_original_rx
#coda19_skewed_rx[toskew_names] <- apply(coda19_skewed_rx[toskew_names], 2, log)


customlog <-function(x){
  # custom function using the appropriate log transform function according
  # to wheter or not the column contains 0
  if (sum (x <= 0)){
    return(log(x+1))
  }
  else(
    return(log(x))
  )
}
coda19_skewed_rx <- coda19_original_rx%>%dplyr::mutate(across(toskew_names, customlog))
coda19_original_rx0 <- coda19_original_rx0%>%dplyr::mutate(across(toskew_names_rx0, customlog))

# 3) FAMD analysis
# FAMD uses standardization therefore skewed variables must be transformed appropiately

# FAMD on data with imaging

famd.coda19 <- FAMD(coda19_skewed_rx,
                    ncp = 30, # number of dimensions
                    sup.var = c(1,6,22,36,37,38,39), # Removing patient id, wbc, fio2 (highly correlated), death, icu, mv (outcomes) and wave FROM analysis
                    graph = FALSE)

# FAMD on data without imaging

famd.coda19_rx0 <- FAMD(coda19_original_rx0,
                    ncp = 30, # number of dimensions
                    sup.var = c(1,6,22,34,35,36,37), # Removing patient id, wbc, fio2 (highly correlated), death, icu, mv (outcomes) and wave FROM analysis
                    graph = FALSE)



### (S3 TABLE)
kableExtra::kable(as_tibble(famd.coda19$eig, rownames='FAMD comp')%>%head(20), caption = "FAMD Percentage of Variance")%>%
  kableExtra::kable_classic(full_width = F, html_font = "helvetica")




# Comparing various clustering algorithms -----------------------------------------


# Selecting and using only the top 12 PCs
famd.coda19_data <- data.frame(famd.coda19$ind$coord)[,1:14]
# Version without imaging data
famd.coda19_data_rx0 <- data.frame(famd.coda19_rx0$ind$coord)[,1:12]

# see below for the OptCluster approach only for FAMD treated dataset
# method 1 optcluster
test <- optCluster::optCluster(famd.coda19_data, 3:6, clMethods = c("kmeans", "agnes", "pam", "diana"), validation = c('internal'))

### (S2 FIGURE)
aggregPlot(test)

### (S2 TABLE)
print <- summary(test)



# Agglomerative Clustering + Results Analysis -----------------------------


##### Computing AGNES (Agglomerative Clustering) clustering
#called kmeans for sake of formatiing but referring to Agglomerative Clustering
famd.kmeans <- eclust(famd.coda19_data, FUNcluster="agnes", agglomeration_method ='ward.D', k=3)
famd.kmeans.rx0 <- eclust(famd.coda19_data_rx0, FUNcluster="agnes", agglomeration_method ='ward.D', k=3)


# Combining clusters to the dataset for complete dataset
tmp <- coda19_original_rx
tmp$clusters <- famd.kmeans$cluster
results <- tmp
# table(results$clusters)

# Combining clusters to the dataset for dataset without imaging
#Changing clusters 1 and 2 (for the sake of interpretation)
tmp <- coda19_original_rx0
tmp$clusters <- famd.kmeans.rx0$cluster
tmp2 <- tmp%>%mutate(
  clusters = ifelse(clusters == 1, 4, clusters)
)

tmp2 <- tmp2%>%mutate(
  clusters = ifelse(clusters == 2, 1, clusters)
)
tmp2 <- tmp2%>%mutate(
  clusters = ifelse(clusters == 4, 2, clusters)
)
results_rx0 <- tmp2
table(results_rx0$clusters)



# Clusters plotting on FAMD
# Vizualize clusters
# FAMD plot with imaging data
famd.coda19_clusters <- FAMD(results,
                             ncp = 30, # number of dimensions
                             sup.var = c(1,6,22,36,37,38,39,40), # Removing selected variables from clustering analysis as above
                             graph = FALSE)

famd_indplot_byicu <- fviz_famd_ind(famd.coda19_clusters,
                                    geom.ind = "point", # show points only (nbut not "text")
                                    habillage = "clusters", # color by clusters
                                    select.var = list(cos2 = 0.5),
                                    alpha.var = 0,
                                    palette = c("#4b595d","#dc8e47", "#05a4d4", "FFFFF"),
                                    select.ind = list(contrib = 100),
                                    #palette = c("#00AFBB", "#E7B800", "#FC4E07"),
                                    addEllipses = TRUE, # Concentration ellipses
                                    repel = TRUE,
                                    legend.title = "Clusters") +
  labs(title = "FAMDs Individuals Plot", x = "PC1", y = "PC2") +
  envalysis::theme_publish() +
  ggtitle("FAMDS Individuals Plot") +
  theme(plot.title = element_text(hjust = 0.5))

# Figure 2 FAMD plot
show(famd_indplot_byicu)




####### Sensitivity Analysis : Comparing rx vs rx0 ---------------------------------------------


# Using rand index statistic
library(fossil)
rand.index(results$clusters, results_rx0$clusters)
mclust::adjustedRandIndex(results$clusters, results_rx0$clusters)


# Exploring patients misclassified
different_all <- results
different_all$difference <- 0
# Marking cases in which cluster differed with and without imaging
different_all[different_all$clusters!=results_rx0$clusters,]$difference <- 1
# Adding clusters without imaging to the dataframe
different_all$clusters_rx0 <- results_rx0$clusters
# different_all now contains all the information regarding reclassified clusters

# Exploring which clusters are more accurate regarding the mortality of cluster
# Analysing clusters that were reclassified from cluster 1 to 2 or 3 after the removal of imaging data
subdif <- different_all%>%filter(difference==1, clusters==1, clusters_rx0==2|3)%>%select(c('death', 'clusters', 'clusters_rx0', 'difference'))
table(subdif$death)

# Analysing clusters that were reclassified from cluster 2 or 3 to cluster 1 after the removal of imaging data

subdif <- different_all%>%filter(difference==1, clusters==2|3, clusters_rx0==1)%>%select(c('death', 'clusters', 'clusters_rx0', 'difference'))
table(subdif$death)

# different <- results[results$clusters!=results_rx0$clusters,]
# different_rx0 <- results_rx0[results$clusters!=results_rx0$clusters,]
# 

# view indices of missclassified <- names(different$clusters)

# vec <- c(0)
# for (i in seq(1,length(to_convert))){
#   vec[i] <- toString(to_convert[i])
# }
# complete list of indices of misclassified individuals
#vector_diff <- paste0(names(different$clusters), ',')

# Creating FAMD plot to vizualize misclassified individuals
# S3 Figure
famd.coda19_different <- FAMD(different_all,
                              ncp = 30, # number of dimensions
                              sup.var = c(1,36,37,38,39,40, 41), # Removing death, icu , wave and clusters
                              graph = FALSE)


ind_coord <- famd.coda19_different$ind$coord[,1:2]
ind_cos2 <- famd.coda19_different$ind$cos2[,1:2]
colnames(ind_cos2) <- c('cos1', 'cos2')
ind_data <- different_all[,c('clusters_rx0','difference')]

ggplot(cbind(ind_coord, ind_cos2, ind_data)%>%filter(cos1>0.05 | cos2>0.01 ),
       aes(x=Dim.1,y=Dim.2,col=as.factor(clusters_rx0),alpha=as.factor(difference)),
       ) + 
  scale_alpha_discrete(range=c(0.3, 1)) +
  scale_colour_manual(values=c("#56B4E9", "#E69F00", "#999999")) +
  geom_point() + 
  theme_bw() +
  labs(col="Clusters (without incorporation \n of imaging data)",
       alpha = 'Misclassified observations')



# Sankey Diagram to vizualize cluster stability rx vs rx0

# Option one using googleVis
# plot(getSankey(results_rx0$clusters, results$clusters))
# Figure 6

df <- as_tibble(cbind(results_rx0$clusters, results$clusters))
colnames(df) <- c('Clustering without Imaging', 'Clustering with Imaging')


df <- df%>%
  ggsankey::make_long('Clustering without Imaging', 'Clustering with Imaging')

ggplot(df, aes(x = x, next_x = next_x, node = node, next_node = next_node, fill = factor(node), label = node)) +
  geom_sankey(flow.alpha = .6,
              node.color = "gray30") +
  geom_sankey_label(size = 3, color = "white", fill = "gray40") +
  # ggsci::scale_color_jama() +
  scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9")) +
  theme_sankey(base_size = 18) +
  labs(x = NULL,
       fill = "Clusters"
       ) +
  theme(legend.position = "right",
        plot.title = element_text(hjust = .5)) +
  ggtitle("Clusters Stability")



# Observing Imaging Opacities Size of misclassified patients
# S4 figure

different_all%>%
  ggplot(aes(x=opacities_size, fill=as.factor(difference))) +
  geom_density(alpha=0.3) +
  ggsci::scale_color_jama() +
  #ggsci::scale_fill_jama() +
  #facet_grid(clusters_rx0 ~ .  , scales = "fixed") +
  facet_wrap(~clusters_rx0, nrow = 1) + #ncol
  # geom_vline(data=mu, aes(xintercept=grp.mean, color=as.factor(clusters)),
  #            linetype="dashed", show_guide = FALSE) +
  #theme_minimal() +
  envalysis::theme_publish() +
  labs(fill = "Reclassified", color = "Mean") +
  xlab("Opacities Size") +
  ylab("Density")



different_all$clusters_rx0



# Analyzing Results - Tabular + Plot ------------------------------------------------

# Creating list of variables
# All variables
alllist <- names(results)
# Categorical variables only
catlist <- names(results[sapply(results, is.factor)])


# Tabular vizualization of clusters without preprocessing
results_famd <- results
# Adding LOS
results_famd <- results_famd%>%left_join(coda_los, by = "patient_site_uid")

### Table 1 (baseline characteristics)
tableOne_baseline <- tableone::CreateTableOne(vars = alllist, data = results_famd, factorVars = catlist, test=TRUE)

# Comparing results according to clusters using the taleone package

tableOne_compared <- tableone::CreateTableOne(vars = alllist, strata="clusters", data = results_famd, factorVars = catlist, test=TRUE)
tableone::kableone(tableOne_compared, nonnormal = c(toskew_names, 'dbp', 'los'))%>%
  kableExtra::kable_classic(full_width = F, html_font = "Helvetica")
#### Table 3 
print(tableOne_compared, nonnormal = c(toskew_names, 'dbp'), quote='TRUE')


# binned <- binning(results_famd)
# test <- filter_preplot(binned)


###### Generating plots allowing easy viz of clusters 

# Transforming factors into numeric for mean and scaling
# We are treating categorical variables as continuous here
indx <- sapply(results_famd, is.factor)
as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}
results_famd[indx] <- lapply(results_famd[indx], as.numeric.factor)

# Creating normalizing function
normalize <- function(x, na.rm = TRUE) {
  return((x- min(x)) /(max(x)-min(x)))
}


# # # Calculating mean -> scaling after -> this allows to better see the differences between clusters
radar.data <- results_famd%>%
  #mutate_each(funs(scales::rescale(x=., to = c(0.5,1))), - clusters)%>%
  group_by(clusters) %>% summarise_all(mean)%>%
  mutate_each(funs(scales::rescale(x=., to = c(0.5,1))), - clusters)

# Scaling first -> mean after -> disadvantage, gap is more subtle and ranges is different across the different variables
radar.data2 <- results_famd%>%
  # Scaling data using normalization
  dplyr::mutate(across(toskew_names, customlog))%>%
  mutate(across(c(1:39), ~ normalize(.x)))%>%
  group_by(clusters) %>% 
  summarise_all(mean)

# Selecting the most important variables
# Calculating the variance first and selecting top 7 variables
variance <- radar.data2%>%summarise_at(c(1:40), ~ var(.x))%>%select(-c('clusters', 'patient_site_uid', 'bronchodilator_agents', 'factor_xa_inhibitors',
                                                                       'diuretics', 
                                                                       'antihypertensive_agents', 'anticholesteremic_agents', 'hypoglycemic_agents', 
                                                                       'platelet_aggregation_inhibitors',
                                                                       'mv', 'wave', 'icu', 'death'))%>%melt()
variance <- variance%>%arrange(desc(value))%>%head(10)
topvar <- unlist(str_split(str_trim(toString(variance$variable)), ', '))
topvar <- c(topvar, 'shock_index')
# topvar contains the 7 variables with the most interclusters variation
# those variables will be used for the barplot
topvar <- topvar[c(5, 4, 11, 8, 9, 1, 2, 3, 6, 7, 10)]

# wide to long
melted <- radar.data2%>%select(topvar, clusters)%>%mutate(id=row_number())%>%melt(id.vars = c('id', 'clusters'), measure.vars = topvar)


### Viz 1 with bar plot = figure 3b
barplot_clusters <- melted%>%ggplot(aes(x=variable, y=value, fill=as.factor(clusters))) + #  after_stat(density)
  geom_col(position = "dodge") +
  ggsci::scale_color_jama() +
  ggsci::scale_fill_jama() +
  #facet_grid(clusters ~ .  , scales = "free") +
  #facet_wrap(~clusters, nrow = 1) + #ncol
  #theme_minimal() +
  envalysis::theme_publish() +
  labs(fill = "Clusters", color = "Mean") +
  xlab("Variables") +
  ylab("Scaled Value")


### Viz 2 with radar plot = figure 3a

create_beautiful_radarchart <- function(data, color = "#00AFBB", 
                                        vlabels = colnames(data), vlcex = 0.7,
                                        caxislabels = TRUE, title = "COVID-19 Clusters", ...){
  fmsb::radarchart(
    data, axistype = 0,
    # Customize the polygon
    pcol = color, pfcol = scales::alpha(color, 0.3), plwd = 1, plty = 1,
    # Customize the grid
    cglcol = "grey", cglty = 2, cglwd = 0.8,
    # Customize the axis
    axislabcol = "black", 
    # Variable labels
    vlcex = vlcex, 
    vlabels = vlabels,
    caxislabels = caxislabels, 
    title = title
  )
  
}

# Prepping the data for radar plot
radar_prep <- radar.data%>%select(topvar, clusters)
min = rep(0.3,ncol(radar_prep))
max = rep(1.2,ncol(radar_prep))
radar_prep <- rbind(max, min, radar_prep) 

create_beautiful_radarchart(radar_prep[c(1,2,3,4,5), 1:11], caxislabels = c(0, 1),
                            color = c("#4b595d","#dc8e47", "#05a4d4"))
# cluster 1 color 1 black
# cluster 2 orange
# cluster 3 blue

legend(x="topright", y=0, 
       legend = paste("Cluster", 1:3),  #rownames(radar_prep[-c(1,2),10]
       bty = "n", pch=20 , 
       col= c("#4b595d","#dc8e47", "#05a4d4"), 
       text.col = "black", 
       cex=1, 
       pt.cex=1)


# Other complementary analysis --------------------------------------------

### Random forest feature importance 

fit_rf = randomForest(factor(results_famd[[40]]) ~., data=results_famd[,2:35])
print(as.data.frame(randomForest::importance(fit_rf))%>%arrange(desc(MeanDecreaseGini)))



# S1 figure
varImpPlot(fit_rf,type=2)



### CART training
### Decision Tree
library(rpart)
library(party)
library(rattle)

# Simple Cart
results_famd <- results

set.seed(42)

# Training CART algorithm
cart_famd <- results_famd
cart_famd$clusters <- as.factor(cart_famd$clusters)
levels(cart_famd$clusters) <- c("Cluster 1", "Cluster 2", "Cluster 3")
cart_famd$clusters <- factor(cart_famd$clusters, labels = c("Cluster 1", "Cluster 2", "Cluster 3"))


fit_cart_all <- rpart(factor(cart_famd[[40]]) ~., 
                  data=cart_famd[,c(2:4, 7:19, 21, 23:35)], #removing wbc, fio2 and clinical outcomes from analysis
                  cp = 0.02)

# without opacities_size
#fit_cart_rx0 <- rpart(factor(results_rx0[[38]]) ~., data=results_rx0[,c(2:33)])
# fit_cart_selected <- rpart(factor(clusters) ~ patient_age + s_f + nlr + mci, 
#                            cp = 0.02,
#                            data=cart_famd)




# Modifying CART plot legend
set.seed(50)
split.fun <- function(x, labs, digits, varlen, faclen)
{
  # replace variable names in the labels
  labs   <- sub("s_f",   "SaO2/FiO2", labs)
  # labs <- sub("survived", "survived", labs)
  labs <- sub("opacities_size",      "Opacities Size", labs)
  labs   <- sub("patient_age",      "Age", labs)
  labs   <- sub("mci",    "MCI", labs)
  labs   <- sub("nlr",    "NLR", labs)
  labs   <- sub("clusters",    "Clusters", labs)
  labs # return the modified labels
}


# Plotting using the rpart package
# rpart.plot::rpart.plot(fit_cart_selected,
#                        box.palette = list('#9aa8ab','#dc8e47','#05a4d4'),
#                        split.fun=split.fun,
#                        type=2, 
#                        extra=8, 
#                        nn=FALSE, 
#                        yesno = 2,
#                        split.border.col=1.,
#                        fallen.leaves=TRUE, 
#                        #roundint = TRUE,
#                        faclen=0, 
#                        varlen=0, 
#                        shadow.col="grey", 
#                        branch.lty=3)


### Figure 5
rpart.plot::rpart.plot(fit_cart_all,
                       box.palette = list('#9aa8ab','#dc8e47','#05a4d4'),
                       split.fun=split.fun,
                       type=2, 
                       extra=8, 
                       nn=FALSE, 
                       yesno = 2,
                       split.border.col=1.,
                       fallen.leaves=TRUE, 
                       #roundint = TRUE,
                       faclen=0, 
                       varlen=0, 
                       shadow.col="grey", 
                       branch.lty=3)


rpart.plot::rpart.rules(fit_cart)


# # Decision Tree without Imaging
# rpart.plot::rpart.plot(fit_cart_rx0,
#                        split.fun=split.fun,
#                        type=2, extra=8, nn=TRUE, 
#                        yesno = 2,
#                        split.border.col=1.,
#                        fallen.leaves=TRUE, 
#                        #roundint = FALSE,
#                        faclen=0, 
#                        varlen=0, 
#                        shadow.col="grey", 
#                        branch.lty=3)


# rpart.plot::rpart.plot(fit_cart, split.fun=split.fun,
#                        roundint=TRUE,
#                        type=4, extra=8, nn=TRUE, fallen.leaves=TRUE, 
#                        faclen=0, varlen=0, shadow.col="grey", branch.lty=3)

# rpart.plot::rpart.plot(fit_cart, split.fun=split.fun,
#                        type=5, extra=8, nn=TRUE, fallen.leaves=TRUE, 
#                        faclen=0, varlen=0, shadow.col="grey", branch.lty=3)

# rpart.plot::rpart.plot(fit_cart, type=2, extra=104, nn=TRUE, fallen.leaves=TRUE,
#                        faclen=0, varlen=0, shadow.col="grey", branch.lty=3)





# In absolute number
# rpart.plot::rpart.plot(fit_cart, type = 2, fallen.leaves = TRUE, extra = 2)
# Proportions of all the clusters
#rpart.plot::rpart.plot(fit_cart, type = 2, fallen.leaves = TRUE, extra = 4)

# Baseline Characteristics ------------------------------------------------

coda19_raw <- read_csv("./datasets/covid24h_notimputed.csv")
transform_data <- function(coda19) {
  coda19 <- coda19%>%mutate_if(function(x){(length(unique(x)) < 9)}, function(x){as.factor(x)})
  coda19 <- coda19%>%mutate_if(function(x){(length(unique(x)) >= 9)}, function(x){as.numeric(x)})
  return (coda19)
}

# S1 table
coda19_raw <- transform_data(coda19_raw)
tableOne_raw <- tableone::CreateTableOne(data = coda19_raw, includeNA = TRUE, test= FALSE)

# Selected variables only
# adding los
# new <- results%>%left_join(coda_los, by = "patient_site_uid")
tableOne_final <- tableone::CreateTableOne(data = new, includeNA = TRUE, test= FALSE)
toskew_names
# 
# print(tableOne_final, missing = TRUE, nonnormal = TRUE, quote = TRUE)
# 
# tableone::kableone(tableOne_raw)%>%
#   kableExtra::kable_classic(full_width = F, html_font = "Helvetica")
# 
# kable(
#   skim(coda19_raw)%>%
#     focus(n_missing, numeric.hist)
#   )%>%kableExtra::kable_classic(full_width = F, html_font = "Helvetica")


# save.image(file = "CODA19_phenotyper_new_2022.RData")

