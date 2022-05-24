# Library load
source("./pkg/library_load.R")

# This script transforms the raw dataset into an imputed set - essentially handling missing data

# Loading raw dataset ---------------------------------------------------------------------
#
# ### Reading episodes data
stay <- read.csv('./data/raw/covid_stay.csv')%>%select(-X)
#  cut-off second wave as 08-2020
#first wave = 0, second wave = 1

#adding month of admission to the dataset
stay <- stay%>%mutate(
  wave = ifelse(lubridate::date(episode_start_time) < lubridate::date("2020-08-01"), 0, 1),
  month = lubridate::month(episode_start_time)
  )%>%distinct(patient_site_uid, .keep_all = TRUE)%>%
  select(c("patient_site_uid", "wave", "month"))

# ### Reading the CSV script
files_path <- list.files(path = "./data/processed", full.names=TRUE)
files_names <- as_tibble(str_split_fixed(list.files(path = "./data/processed"), pattern =".csv", n=2))[[1]]
for (i in seq_along(files_path))
  print(paste0('Currently reading ...', files_names[i]))
  #dbGetQuery(coda19, statement = read_file(files_path[i]))
  tmp <- read.csv(files_path[i])
  tmp <- tmp%>%left_join(stay, by = 'patient_site_uid')
  if ("X"%in%colnames(tmp)){
    tmp <- tmp%>%select(-X)
  }
  
  # Outliers removal (removing n = 2)
  # Age 
  tmp <- tmp%>%filter(patient_age < 120 & patient_age > 18)
  # Creatinine - All extremes seemed appropriate, no values changed
  
  # Replacing NA to 0s in order for case_when to execute
  tmp$fio2 <- tmp$fio2%>%replace_na(0)
  
  # FiO2
  # Different correction for patients on MV
  
  tmp <- tmp%>%mutate(
    fio2=
      case_when(
        fio2 == 0 & mv == 1 ~ 40,
        fio2 > 100 ~ 100,
        TRUE ~ fio2))
  
  # Different correction for patients not on MV
  tmp <- tmp%>%mutate(
    fio2=
      case_when(
        fio2 == 0 & mv == 0 ~ 21,
        fio2 > 100 ~ 100,
        TRUE ~ fio2))
  
  # SO2
  tmp <- tmp%>%mutate(
    so2=
      case_when(
        so2 < 40 ~ 60,
        so2 > 100 ~ 100,
        TRUE ~ as.double(so2)
        ))
  
  # Sanity Check
  #table(tmp$so2, useNA="always")
  
  
  # RR
  tmp <- tmp%>%mutate(
    rr=
      case_when(
        rr  < 8 ~ 12,
        rr >= 50 ~ 50,
        TRUE ~ as.double(rr)))
  
  # shock/index
  tmp <- tmp%>%mutate(
    "shock_index" = hr/sbp
  )
  
  # sao2/fio2
  tmp <- tmp%>%mutate(
    "s_f" = so2/(fio2*0.01))
  


# Imputation --------------------------------------------------------------

### NB must manually modify script to generate imputed set at 24, 48 and 72h

#hours = c("24h", "48h", "72h")
hours = c("24h")

#save <- tmp
for (i in hours){
  # Dropping columns with more than 30% missing values  
  # Dropping observations with more than 30% missing variables
  df_name = paste0("covid", i, "_notimputed")
  assign(df_name, tmp)
  tmp <- tmp %>% 
    purrr::discard(~sum(is.na(.x))/length(.x)*100 >=30)
    #filter(rowSums(is.na(.)) < ncol(.)*0.30)
  # Getting predictor Matrix and modifying it to exclude 2 variables from the imputation set 
  matrix <- mice::mice(tmp, method = "cart", m=1, maxit = 0)
  pred_matrix <- matrix$predictorMatrix
  pred_matrix[,'patient_site_uid'] <- 0
  pred_matrix[,'death'] <- 0
  pred_matrix[,'month'] <- 0
  pred_matrix[,'wave'] <- 0
  
  # Single CART Imputation
  covidimputer <- mice::mice(tmp, pred = pred_matrix, method = "cart", m=1)
  imputed_df <- complete(covidimputer, 1)
  
  
  # Adding AG (for original version only)
  # ag
  imputed_df <- imputed_df%>%mutate(
    "anion_gap" = sodium-bicarbonate-chloride)
  
  # # #Saving the file
  write_csv(imputed_df, file=paste0('./data/imputed/','covid',i,'_imputed_original.csv'))
}


