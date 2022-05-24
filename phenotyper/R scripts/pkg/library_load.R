# Function to load or install
install_or_load_pack <- function(pack){
  new.pkg <- pack[!(pack %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pack, require, character.only = TRUE)
}

# Packages used

pack <- c("tidyverse", "bigrquery","plotly","scales","RColorBrewer","data.table","knitr","corrplot","Hmisc","stats", "janitor", "lubridate", "testthat", "magrittr", 
          "rjson", "DBI", "comorbidity", "RSQLite", "fuzzyjoin",
          "purrr", "healthcareai")

# Loading the pack
install_or_load_pack(pack)

