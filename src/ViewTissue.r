library(Seurat)
library(future)
plan("multisession", workers = 10)
library(ggplot2)

path = "/Users/antanas/GitRepo/ChromeX/data"

# Load the Xenium data
xenium.obj <- LoadXenium(path, fov = "fov")
# remove cells with 0 counts
xenium.obj <- subset(xenium.obj, subset = nCount_Xenium > 0)

ImageDimPlot(xenium.obj, fov = "fov", molecules = c("Gad1", "Sst", "Pvalb", "Gfap"), nmols = 20000)

library(infercnv)
counts_matrix = GetAssayData(xenium.obj, slot="counts")
