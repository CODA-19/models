# CODA-19 phenotyper project

### Background

------

This data science project aimed at identifying phenotypes for hospitalized with covid-19 using cluster analysis. The repo contains the entire code needed to reproduce the results.

### Files organization

------



```
├── Python scripts
    └── rxcui_mapper
        └── rxcui_mapper.py
├── R scripts
    ├── coda19_phenotyper.Rproj
    ├── utils_template.R
    ├── data
        ├── imaging
        ├── imputed
        ├── other
        ├── processed
        └── raw
    ├── sql
    ├── src
        ├── 1.generate_processed_data.R
        ├── 2.impute_processed_data.R
        ├── 3.clustering_script.R
        └── 4.survival_analysis.R
    └── pkg
        ├── drug_class_dict_2.csv
				└── library_load.R
				
```

### Python scripts

------

-  **`rxcui_mapper.py`** 
  * Maps the raw drug names contained in the coda-19 db to MESH drug class categories

### R scripts

------

- Run all the enumerated files in order (1 →  4) to generate all the findings reported in the manuscript.
- **`1.generate_processed_data.R`**  loads raw data extracted from the database and wrangles it into a wide format.  Generates **`data/processed/covid24h_notimputed.csv`**. Contains all the candidate variables per observation.
-  **`2.impute_processed_data.R`** handles missing data and imputation.  Generates **`data/imputed/covid24h_imputed_original.csv`**. 
- **`3.clustering_script.R`** contains the code for cluster analysis.
- **`4.survival_analysis.R`** contains the code for survival analysis.

### Data availability

------

- Raw data were generated at the CHUM. Derived data supporting the findings of this study are available from the corresponding author MC on request.
