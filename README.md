# bioinf-proj
Methylation sites as a biomarker for cancer types - Bioinformatics project #4

## Usage
main.py --help for the complete list of commands


usage: main.py [-h] [-cl2 | -cl3]
               [-svm | -knn | -rforest | -kmeans | -hierarc] [-over | -smote]
               [-ttest | -fisher | -anova | -pca | -lda | -sfs | -ga]
               [-d DOWNLOAD DOWNLOAD] [-ds DOWNLOADSANE DOWNLOADSANE]
               [-s STORE] [--alpha ALPHA] [--perc PERC] [-rs R_STATE]
               [--only_chrms_t] [--crossval] [--plot_lc]
               [--remove_nan_cpgs REMOVE_NAN_CPGS]

Bioinf project. The arguments can be passed in any order.

optional arguments: 

  -h, --help            show this help message and exit 
  -cl2                  in order to classify two cancer types. 
  -cl3                  in order to classify two cancer types AND sane. 
  -svm                  train a Support Vector Machine classifier 
  -knn                  train a K Nearest Neighbors classifier 
  -rforest              train a Random Forest classifier 
  -kmeans               train a Kmeans clustering 
  -hierarc              train an Agglomerative Hierarchical clustering
  -over                 imbalance: Random Oversampling
  -smote                imbalance: SMOTE
  -ttest                feature selection: ttest per chromosoma and per cpg
                        site - 2 classes
  -fisher               feature selection: fisher criterion - 3 classes
  -anova                feature selection: anova - 3 classes
  -pca                  dimensionality reduction: Principal Component Analisys
  -lda                  dimensionality reduction: Linear Discriminant Analysis
  -sfs                  feature selection - wrapper: Step Forward Selection
                        (nearly unfeasible)
  -ga                   feature selection - wrapper: Genetic Algorithm
  -d DOWNLOAD DOWNLOAD, --download DOWNLOAD DOWNLOAD
                        download Adenoma and Adenocarcinoma and Squamous Cell
                        Neoplasm data from Genomic Data Common. It needs 2
                        parameters: first parameter is the destination folder;
                        second parameters is the number of files to be
                        downloaded for each class
  -ds DOWNLOADSANE DOWNLOADSANE, --downloadsane DOWNLOADSANE DOWNLOADSANE
                        download Sane data from Genomic Data CommonIt needs 2
                        parameters: first parameter is the destination folder;
                        second parameters is the number of files to be
                        downloaded
  -s STORE, --store STORE
                        concatenate files belonging to same cancer type and
                        store them in a binary file
  --alpha ALPHA         to set a different ALPHA: ttest parameter - default is
                        0.001
  --perc PERC           to set PERC of varaince explained by the features kept
                        by PCA
  -rs R_STATE, --r_state R_STATE
                        to set a user defined Random State - default is 8
  --only_chrms_t        select only chrms for ttest
  --crossval            to do crossvalidation OR in case of unsupervised to
                        plot the Inertia curve
  --plot_lc             plot the learning curve
  --remove_nan_cpgs REMOVE_NAN_CPGS
                        IF True: removes features containing at least one NaN
                        value. IF False: NaN are substituted by the mean over
                        the feature. The old file resulted by feature
                        reduction must be eliminated when changing option. By
                        Default is True.



## Interesting GDC Queries

***Adenomas and Adenocarcinomas***

cases.disease_type in ["Adenomas and Adenocarcinomas"]  
and cases.primary_site in ["Bronchus and lung"]  
and files.data_category in ["DNA Methylation"]  
and files.platform in ["Illumina Human Methylation 450"]  
and cases.samples.sample_type in ["Primary Tumor","Recurrent Tumor"] 

**files***: 437  

***cases***: 425  

***size***: 61.73 GB  


***Squamous Cell Neoplasm***

cases.disease_type in ["Squamous Cell Neoplasms"]  
and cases.primary_site in ["Bronchus and lung"]  
and files.data_category in ["DNA Methylation"]  
and files.platform in ["Illumina Human Methylation 450"]  
and cases.samples.sample_type in ["Primary Tumor","Recurrent Tumor"] 

***files***: 370  

***cases***: 372  

***size***: 52.28 GB  


## Methylation Liftover Fields

* Composite Element: A unique ID for the array probe associated with a CpG site  
* Beta Value: Represents the ratio between the methylated array intensity and total array intensity,  
  falls between 0 (lower levels of methylation) and 1 (higher levels of methylation)  
* Chromosome: The chromosome in which the probe binding site is located  
* Start: The start of the CpG site on the chromosome  
* End: The end of the CpG site on the chromosome  
* Gene Symbol: The symbol for genes associated with the CpG site. 
  Genes that fall within 1,500 bp upstream of the transcription start site (TSS) to the end of the gene body are used.  
* Gene Type: A general classification for each gene (e.g. protein coding, miRNA, pseudogene)  
* Transcript ID: Ensembl transcript IDs for each transcript associated with the genes detailed above  
*Position to TSS: Feature Type Distance in base pairs from the CpG site to each associated transcript’s start site  
* CGI Coordinate: The start and end coordinates of the CpG island associated with the CpG site  
* Feature Type: The position of the CpG site in reference to the island: 
  Island, N_Shore or S_Shore (0-2 kb upstream or downstream from CGI), or N_Shelf or S_Shelf (2-4 kbp upstream or downstream from CGI)  
