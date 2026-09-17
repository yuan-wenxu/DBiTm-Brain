script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) {
  stop("Cannot determine the path of 00-loadmethylation.R.", call. = FALSE)
}
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
script_dir <- dirname(script_path)
source(file.path(script_dir, "dbit-io.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop(
    paste0(
      "Usage: Rscript <path/to/00-loadmethylation.R> ",
      "<matrix_root> <output.qs>"
    ),
    call. = FALSE
  )
}

spot_radius_um <- 50
overwrite <- FALSE

matrix_root <- normalizePath(args[[1L]], mustWork = TRUE)
output_path <- path.expand(args[[2L]])

if (file.exists(output_path) && !overwrite) {
  message("Output already exists; leaving it unchanged: ", output_path)
  quit(save = "no", status = 0L)
}

methylation <- LoadDBiTMethylation(
  data_path = matrix_root,
  spot_radius = spot_radius_um
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
qs::qsave(methylation, output_path)
message(
  "Saved ", ncol(methylation), " spots x ", nrow(methylation),
  " features to ", output_path
)
