script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) {
  stop("Cannot determine the path of 00-loadmRNA.R.", call. = FALSE)
}
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
script_dir <- dirname(script_path)

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  source(file.path(script_dir, "dbit-io.R"))
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop(
    "Usage: Rscript <path/to/00-loadmRNA.R> <matrix_path> <output.qs>",
    call. = FALSE
  )
}

spot_radius_um <- 50
overwrite <- FALSE

matrix_path <- normalizePath(args[[1L]], mustWork = TRUE)
output_path <- path.expand(args[[2L]])
output_dir <- dirname(output_path)

if (file.exists(output_path) && !overwrite) {
  message("Output already exists; loading it for plotting: ", output_path)
  mrna <- qs::qread(output_path)
} else {
  mrna <- LoadDBiT(
    data_path = matrix_path,
    assay = "Spatial",
    spot_radius = spot_radius_um
  )

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  qs::qsave(mrna, output_path)
  message(
    "Saved ", ncol(mrna), " spots x ", nrow(mrna), " genes to ", output_path
  )
}

image_name <- Images(mrna)[[1L]]
image <- mrna[[image_name]]
p <- SpatialFeaturePlot(
  mrna,
  features = "nCount_Spatial",
  shape = 22,
  pt.size.factor = 2,
  crop = TRUE,
  max.cutoff = "q95"
) +
  coord_fixed(
    xlim = c(0, ncol(image)),
    ylim = c(0, nrow(image)),
    expand = FALSE,
    ratio = 1
  ) +
  theme(legend.position = "right")

plot_path <- file.path(
  output_dir,
  paste0(tools::file_path_sans_ext(basename(output_path)), "_nUMIs.png")
)
ggsave(plot_path, p, width = 5, height = 4, dpi = 300)
message("Saved nUMI plot to ", plot_path)
