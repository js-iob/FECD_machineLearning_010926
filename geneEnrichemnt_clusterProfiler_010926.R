BiocManager::install("clusterProfiler")
BiocManager::install("org.Hs.eg.db")

library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)

setwd('D:\\IOB\\Projects\\ocularDiseases_ML\\clusterProfiler\\FECD\\nr_list')
gene_symbols <- read.table("nr_geneList_svm_lr.txt", header = FALSE, stringsAsFactors = FALSE)$V1
length(gene_symbols)    

gene_ids <- bitr(gene_symbols,
                 fromType = "SYMBOL",
                 toType   = "ENTREZID",
                 OrgDb    = org.Hs.eg.db)
print("Symbol-to-Entrez mapping:")
print(gene_ids)


ego <- enrichGO(gene          = gene_ids$ENTREZID,
                OrgDb         = org.Hs.eg.db,
                keyType       = "ENTREZID",
                ont           = "BP",
                pAdjustMethod = "BH",
                qvalueCutoff  = 0.05,
                readable      = TRUE)
write.csv(ego@result, file = "FECD_ego_results.csv", row.names = FALSE)

head(ego@result$Description, 20)
ego_df <- ego@result
go_terms <- c(
  'oxidative phosphorylation',
  'positive regulation of MAPK cascade',
  'epithelial to mesenchymal transition',
  'extracellular matrix organization',
  'positive regulation of inflammatory response',
  'response to oxidative stress',
  'endothelial cell migration'
)
ego_df_sub <- ego@result[ego@result$Description %in% go_terms, ]
valid_ids <- intersect(ego_df_sub$ID, names(ego@geneSets))

ego_sub <- new(
  "enrichResult",
  result        = ego@result[ego@result$ID %in% valid_ids, ],
  pvalueCutoff  = ego@pvalueCutoff,
  pAdjustMethod = ego@pAdjustMethod,
  qvalueCutoff  = 1,
  organism      = ego@organism,
  ontology      = ego@ontology,
  keytype       = ego@keytype,
  gene          = ego@gene,
  universe      = ego@universe,
  geneSets      = ego@geneSets[valid_ids],
  readable      = FALSE
)

class(ego_sub)
pdf("cnetplot_colored.pdf", width = 30, height = 20)
cnetplot(ego_sub, showCategory = 7, color_edge = 'category')
edges <- p$data[, c("category", "gene")]
dev.off()

#Creating upsetplot
pdf("GO_upsetmap.pdf", width = 30, height = 20)
upsetplot(ego_sub)
dev.off()



#KEGG pathway enrichment
ekegg <- enrichKEGG(gene         = gene_ids$ENTREZID,
                    organism     = "hsa",
                    pvalueCutoff = 0.05)
ekegg <- setReadable(ekegg, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")

write.csv(ekegg@result, file = "KEGG_results.csv", row.names = FALSE)
ekegg_df <- ekegg@result
ekegg_df_sub <- ekegg_df[ekegg_df$Description %in% c(
  'PI3K-Akt signaling pathway','ECM-receptor interaction','MAPK signaling pathway','Calcium signaling pathway','p53 signaling pathway'), ]
ekegg_sub <- new("enrichResult",
                 result        = ekegg_df_sub,
                 pvalueCutoff  = ekegg@pvalueCutoff,
                 pAdjustMethod = ekegg@pAdjustMethod,
                 qvalueCutoff  = ekegg@qvalueCutoff,
                 organism      = ekegg@organism,
                 ontology      = ekegg@ontology,
                 keytype       = ekegg@keytype,
                 gene          = ekegg@gene,
                 universe      = ekegg@universe,
                 geneSets      = ekegg@geneSets,
                 readable      = ekegg@readable)
class(ekegg_sub)
pdf("cnetplot_colored_kegg.pdf", width = 30, height = 20)
cnetplot(ekegg_sub, showCategory = 10, color_edge = 'category')
dev.off()
dotplot(ekegg_sub, showCategory = 15)

pdf("ekegg_barplot_091525.pdf", width = 25, height = 20)
barplot(ekegg_sub, showCategory = 10)
dev.off()
