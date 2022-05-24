# Set Working Directory

# Library load
source("./pkg/library_load.R")

# # ### Reading the CSV script
files_path <- list.files(path = "./data/raw/", full.names=TRUE)
files_names <- as.tibble(str_split_fixed(list.files(path = "./data/raw/"), pattern =".csv", n=2))[[1]]
for (i in seq_along(files_path)){
  print(paste0('Currently reading ...', files_names[i]))
  #dbGetQuery(coda19, statement = read_file(files_path[i]))
  tmp <- read.csv(files_path[i])
  if ("X"%in%colnames(tmp)){
    tmp <- tmp%>%select(-X)
  }
  assign(files_names[i], tmp)
}

# Data wrangling --------------------------------------------------------


# Transforming narrow dataframe to wide - only 3 : comorbidities, dx and drugs


# 1) Comorbidities using the Comorbidity package
covid_comorbidities <- comorbidity::comorbidity(x = covid_comorbidities, id = "patient_site_uid", code = "diagnosis_icd_code", score = "charlson", icd = "icd10", assign0 = TRUE)
covid_comorbidities <- covid_comorbidities%>%select(-c("index", "wscore", "windex"))

# 2 ) Drugs

# covid_demographics <- read.csv('./csv/covid_demographics.csv')%>%select(-X)

# Function that cadds count of each dx in the

# The script was changed count=n() to simply get 1 if a drug is recorded at any time regardless if more than once
get_counts <-function(dataset){       
  summary <- dataset %>% group_by(patient_site_uid,drug_class) %>% dplyr::summarise(count=1)%>% arrange(desc(count))%>%ungroup(patient_site_uid) 
  return(summary)
}
# Reading the dictionary
drug_dict <- read.csv('./pkg/drug_class_dict_2.csv')%>%select(-X)
drug_dict$drug_name <- stringr::str_replace_all(drug_dict$drug_name, "(\\*|\\#|:|\\(|\\)|\\.|\\,|\\/|\\%)", " ")

# Formatting all the drugs data
# drugs_list = list("covid_drugs24h" = covid_drugs24h, "covid_drugs48h" = covid_drugs48h, "covid_drugs72h" = covid_drugs72h)
covid_drugs24h$drug_name <- stringr::str_to_lower(covid_drugs24h$drug_name)
drugs_list = list("covid_drugs24h" = covid_drugs24h)
#f <- drugs_list[[1]]
j = 1
for (i in drugs_list){
  i$drug_name <- stringr::str_remove_all(i$drug_name, "(\\*|\\#|:|\\(|\\)|\\.|\\,|\\/|\\%)")
  i$drug_name <- stringr::str_remove_all(i$drug_name, "([0-9]|\\bmg\\b|\\bco\\b|\\binj\\b|\\bml\\b|\\bxr\\b|\\buml\\b|\\bmgh\\b|\\binj|\\bco-h|g\\b|\\bmcginh\\b|\\bsupp\\b|\\bmmol\\b|\\buml|\\bmcg\\b|\\bmgml\\b)")
  i$drug_name <- stringr::str_squish(i$drug_name)
  # Replacing meds to match new format
  i$drug_name <- stringr::str_replace(i$drug_name, "(acide acetylsalicylique|aas(.*))", "aspirin")
  i <- i%>%filter(drug_name!='')
  #tmp <- i%>%fuzzyjoin::regex_left_join(drug_dict, by=c('drug_name'='drug_name', ignore_case = TRUE))%>%
    #select(-c('drug_name', 'drug_start_time'))
  i$drug_regex <- gsub(" ", "|", i$drug_name)
  tmp <- i%>%fuzzyjoin::regex_left_join(drug_dict, by=c('drug_regex'='drug_name'))%>%
    select(c('patient_site_uid', 'drug_class'))%>%
    drop_na() # Removes all umatched drugs - those drugs are usually not clinical relevant drugs
    #mutate(dist = stringdist::stringdist(drug_name.x, drug_name.y)) %>% # add distance
    # group_by(patient_site_uid, drug_name.x) %>%                                            # group entries
    # mutate(exact = any(dist == 0)) %>%                               # check if exact match exists in group
    # filter(dist == 0) %>%                                   # keep only entries where no exact match exists in the group OR where the entry is the exact match
    # ungroup()
    # select(-c('drug_name', 'drug_start_time'))
  narrowtmp <- get_counts(tmp)
  widecovid <- narrowtmp%>%spread(drug_class, count, fill=0)%>%select(-"0")
  assign(names(drugs_list)[j], widecovid)
  j = j+1
}



# 3) Collecting info on hospital stay, icu stay and death
covid_stay <- covid_stay%>%select(patient_site_uid)
covid_deaths <- covid_deaths%>%mutate(death=1)%>%select(-death_time)
covid_icustay <- covid_icustay%>%mutate(icu=1)%>%select(c(patient_site_uid, icu))

# 4) Transforming narrow to wide format 

# Demographics 1-hot encoding & keeping age in the dataset
covid_demographics <- covid_demographics%>%mutate(patient_sex = ifelse(as.character(patient_sex) == 'unspecified', 'female', as.character(patient_sex)))
covid_age <- covid_demographics%>%select(c(patient_site_uid, patient_age))
#1 hot encoding
covid_demographics <- reshape2::dcast(data = covid_demographics, patient_site_uid ~ patient_sex, length)
covid_demographics <- covid_demographics%>%left_join(covid_age, by = "patient_site_uid")

covid_comorbidities <- covid_comorbidities #already wide

covid_drugs24h <- covid_drugs24h #already wide
covid_labs24h <- covid_labs24h #already wide

# adding NLR
covid_labs24h$nlr <- covid_labs24h$neutrophil/covid_labs24h$lymph
  
covid_vitals24h <- covid_vitals24h #already wide

# Adding appropriate info on mechanical ventilation
covid_mv24h <- covid_mv24h%>%mutate(mv=1)%>%select(c(patient_site_uid, mv))
# covid_mv48h <- covid_mv48h%>%mutate(mv=1)%>%select(c(patient_site_uid, mv))
# covid_mv72h <- covid_mv72h%>%mutate(mv=1)%>%select(c(patient_site_uid, mv))


# Adding imaging data info
# Using only the list of patients
covid_imaging24h <- covid_imaging24h%>%distinct(patient_site_uid)

# Adding the imaging clustering info

### 5) Generating final COVID database at 24, 48 and 72h---------
# covid_24h <- plyr::join_all(tablelist, by='patient_site_uid', type='left') same script using plyr


tablelist_24h = list(covid_imaging24h, covid_stay, covid_demographics, covid_deaths, covid_comorbidities, covid_drugs24h, covid_labs24h, covid_vitals24h, covid_mv24h, covid_icustay)
covid_24h <- tablelist_24h %>% purrr::reduce(left_join, by = "patient_site_uid", all.x=TRUE)
covid_24h <- covid_24h%>%distinct(patient_site_uid, .keep_all = TRUE)
                                  

# Versions below for data within 48 or 72 hours of admissions, not used.
# tablelist_48h = list(covid_stay, covid_demographics, covid_deaths, covid_comorbidities, covid_drugs48h, covid_labs48h, covid_vitals48h, covid_mv48h, covid_icustay)
# covid_48h <- tablelist_48h %>% purrr::reduce(left_join, by = "patient_site_uid", all.x=TRUE)
# 
# tablelist_72h = list(covid_stay, covid_demographics, covid_deaths, covid_comorbidities, covid_drugs72h, covid_labs72h, covid_vitals72h, covid_mv72h, covid_icustay)
# covid_72h <- tablelist_72h %>% purrr::reduce(left_join, by = "patient_site_uid", all.x=TRUE)

### Saving raw dataset at 24, 48 and 72h----------


# Replacing NULL values by actual real values we know
covid_24h <- covid_24h%>%select(-c('NA'))
covid_24h$death <- replace_na(covid_24h$death, 0) 
covid_24h$mv <- replace_na(covid_24h$mv, 0) 
covid_24h$icu <- replace_na(covid_24h$icu, 0) 
covid_24h <- janitor::clean_names(covid_24h)

# covid_48h <- covid_48h%>%select(-c(`<NA>`, 'NA'))
# covid_48h$death <- replace_na(covid_48h$death, 0) 
# covid_48h$mv<- replace_na(covid_48h$mv, 0) 
# covid_48h$icu <- replace_na(covid_48h$icu, 0) 
# covid_48h <- janitor::clean_names(covid_48h)
# 
# covid_72h <- covid_72h%>%select(-c(`<NA>`, 'NA'))
# covid_72h$death <- replace_na(covid_72h$death, 0) 
# covid_72h$mv<- replace_na(covid_72h$mv, 0) 
# covid_72h$icu <- replace_na(covid_72h$icu, 0) 
# covid_72h <- janitor::clean_names(covid_72h)

# Saving the non-imputed data
write_csv(covid_24h, file='./data/processed/covid24h_notimputed.csv')
# write_csv(covid_48h, file='./data/processed/covid48h_notimputed.csv')
# write_csv(covid_72h, file='./data/processed/covid72h_notimputed.csv')