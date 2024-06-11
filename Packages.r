install.packages('Seurat')
setRepositories(ind = 1:3, addURLs = c('https://satijalab.r-universe.dev', 'https://bnprks.r-universe.dev/'))
install.packages(c("BPCells", "presto", "glmGamPoi"))

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
install.packages('Signac')

if(!requireNamespace("BiocManager", quietly = TRUE)){
     install.packages("BiocManager")
}

remotes::install_github("satijalab/seurat-data", quiet = TRUE)
remotes::install_github("satijalab/azimuth", quiet = TRUE)
remotes::install_github("satijalab/seurat-wrappers", quiet = TRUE)


# install.packages('/Users/antanas/GitRepo/ChromeX/packages/JAGS-4.3.2.pkg', repos=NULL, type="source")
BiocManager::install("infercnv")
