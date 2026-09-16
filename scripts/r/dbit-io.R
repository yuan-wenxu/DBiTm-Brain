#' Read the width and height from a PNG file header
#'
#' Reads only the first 24 bytes of a PNG file and extracts the image width and
#' height from the IHDR chunk. Pixel data are not decoded, so this is suitable
#' for inspecting very large tissue images without allocating memory for the
#' full image.
#'
#' The function also checks the standard eight-byte PNG signature and reports
#' an error when the file is missing, truncated, or is not a valid PNG file.
#'
#' @param path Path to a PNG image. The path must exist and be readable.
#' @return A named numeric vector of length two: `width` and `height`, both in
#'   pixels.
read_png_dimensions <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  con <- file(path, open = "rb")
  on.exit(close(con), add = TRUE)
  header <- readBin(con, what = "raw", n = 24L)
  signature <- as.raw(c(137, 80, 78, 71, 13, 10, 26, 10))
  if (length(header) < 24L ||
      !identical(header[seq_along(signature)], signature)) {
    stop("Not a valid PNG file: ", path, call. = FALSE)
  }
  uint32_be <- function(bytes) sum(as.numeric(bytes) * 256^(3:0))
  c(
    width = uint32_be(header[17:20]),
    height = uint32_be(header[21:24])
  )
}


#' Load DBiT-seq processed data into a Seurat object
#'
#' Reads 10x mtx + tissue_positions, scales the tissue image in memory, and
#' attaches it as VisiumV2.
#'
#' Coordinate layers:
#'   (b) fullres pixels aligned to tissue_raw_image.png — stored in FOV / spatial
#'   (c) lowres display = (b) * tissue_lowres_scalef — used by SpatialFeaturePlot
#'   (d) physical µm = (b) * um_per_px_fullres
#'
#' @param data_path Sample directory (mtx, features, barcodes, tissue_positions, ssDNA).
#' @param image_name Name of the VisiumV2 image slot to create.
#' @param assay Assay name for the expression matrix.
#' @param spot_radius Centroid plot radius in µm.
#' @param um_per_px_fullres µm per fullres pixel for the physical DimReduc.
#' @param fullres_w Fullres ssDNA width. Inferred from the PNG when NULL.
#' @param fullres_h Fullres ssDNA height. Inferred from the PNG when NULL.
#' @param image_scale ImageMagick resize geometry, for example `"20%"`.
#' @param verbose Print progress messages.
#' @return A Seurat object with Spatial assay, spatial/physical DimReduc, and VisiumV2 image.
#'   Pixel size is stored as object[[image]]@misc$um_per_px_fullres.
LoadDBiT <- function(
  data_path,
  image_name = "ssDNA",
  assay = "Spatial",
  spot_radius,
  um_per_px_fullres = 0.294,
  fullres_w = NULL,
  fullres_h = NULL,
  image_scale = "20%",
  verbose = TRUE
) {
  data_path <- normalizePath(data_path, mustWork = TRUE)
  image_path <- file.path(data_path, "tissue_raw_image.png")
  pos_file <- file.path(data_path, "tissue_positions.tsv.gz")

  if (!file.exists(image_path)) {
    stop("Missing ssDNA image: ", image_path, call. = FALSE)
  }
  if (!file.exists(pos_file)) {
    stop("Missing tissue_positions: ", pos_file, call. = FALSE)
  }

  raw_dimensions <- read_png_dimensions(image_path)
  if (is.null(fullres_w)) fullres_w <- raw_dimensions[["width"]]
  if (is.null(fullres_h)) fullres_h <- raw_dimensions[["height"]]

  ## 1. counts ====
  if (isTRUE(verbose)) {
    message("Reading 10x matrix from ", data_path)
  }
  counts <- Seurat::Read10X(data_path)
  seu <- SeuratObject::CreateSeuratObject(counts = counts, assay = assay)

  ## 2. coordinates (DimReduc; (b) fullres) ====
  coords_df <- read.delim(pos_file, row.names = 1)
  missing <- setdiff(SeuratObject::Cells(seu), rownames(coords_df))
  if (length(missing) > 0L) {
    stop(
      length(missing), " barcodes lack tissue_positions coordinates (e.g. ",
      missing[[1]], ").",
      call. = FALSE
    )
  }

  seu$in_tissue <- coords_df[SeuratObject::Cells(seu), "in_tissue"]
  seu$array_row <- coords_df[SeuratObject::Cells(seu), "array_row"]
  seu$array_col <- coords_df[SeuratObject::Cells(seu), "array_col"]

  coords <- as.matrix(
    coords_df[
      SeuratObject::Cells(seu),
      c("pxl_col_in_fullres", "pxl_row_in_fullres")
    ]
  )
  colnames(coords) <- paste0("coords_", 1:2)

  seu[["spatial"]] <- SeuratObject::CreateDimReducObject(
    embeddings = coords,
    key = "coords_",
    assay = assay
  )

  ## 3. image ====
  if (isTRUE(verbose)) {
    message("Reading and scaling ssDNA image ", image_path, " to ", image_scale)
  }
  image_object <- magick::image_read(image_path)
  image_object <- magick::image_scale(image_object, geometry = image_scale)
  ssDNA <- magick::image_data(image_object, channels = "gray")[1, , ]
  ssDNA <- t(ssDNA)
  storage.mode(ssDNA) <- "double"
  ssDNA <- ssDNA / 255
  rm(image_object)

  tissue_lowres_scalef <- dim(ssDNA)[2] / as.integer(fullres_w)
  if (isTRUE(verbose)) {
    message(sprintf(
      "tissue_lowres_scalef (w/h) = %.6f / %.6f",
      tissue_lowres_scalef, dim(ssDNA)[1] / as.integer(fullres_h)
    ))
  }

  spot_radius_px <- as.numeric(spot_radius) / um_per_px_fullres
  if (isTRUE(verbose)) {
    message(sprintf(
      "spot_radius = %.3f µm (%.3f fullres px)",
      spot_radius_px * um_per_px_fullres, spot_radius_px
    ))
  }
  sf <- Seurat::scalefactors(
    spot = spot_radius_px,
    fiducial = 100,
    hires = tissue_lowres_scalef,
    lowres = tissue_lowres_scalef
  )

  coord_img <- data.frame(
    imagecol = coords_df[SeuratObject::Cells(seu), "pxl_col_in_fullres"],
    imagerow = coords_df[SeuratObject::Cells(seu), "pxl_row_in_fullres"],
    row.names = SeuratObject::Cells(seu)
  )
  fov <- SeuratObject::CreateFOV(
    coord_img[, c("imagecol", "imagerow")],
    type = "centroids",
    radius = sf[["spot"]],
    assay = assay,
    key = paste0(image_name, "_")
  )
  seu[[image_name]] <- methods::new(
    Class = "VisiumV2",
    boundaries = fov@boundaries,
    molecules = fov@molecules,
    assay = fov@assay,
    key = fov@key,
    image = ssDNA,
    scale.factors = sf,
    coords_x_orientation = "horizontal",
    misc = list(um_per_px_fullres = um_per_px_fullres)
  )
  SeuratObject::DefaultAssay(seu) <- assay

  ## 4. physical µm ====
  coords_um <- as.matrix(
    SeuratObject::GetTissueCoordinates(seu, image_name)[, c("x", "y")] *
      um_per_px_fullres
  )
  colnames(coords_um) <- c("physical_1", "physical_2")
  rownames(coords_um) <- SeuratObject::Cells(seu)
  seu[["physical"]] <- SeuratObject::CreateDimReducObject(
    embeddings = coords_um,
    key = "physical_",
    assay = assay
  )

  if (isTRUE(verbose)) {
    message(sprintf(
      "Loaded %d spots x %d features; image='%s'",
      ncol(seu), nrow(seu), image_name
    ))
  }
  seu
}


.read_single_dbit_matrix <- function(data_path, matrix_name, verbose = TRUE) {
  if (isTRUE(verbose)) {
    message("Reading ", matrix_name, " matrix from ", data_path)
  }
  matrix <- Seurat::Read10X(data.dir = data_path)
  matrix
}


.parse_dbit_matrix_barcodes <- function(barcodes) {
  valid <- grepl("^[0-9]{2}_?[0-9]{2}$", barcodes)
  if (any(!valid)) {
    invalid <- barcodes[which(!valid)[[1L]]]
    stop(
      "Invalid matrix barcode '", invalid,
      "'; expected RRCC or RR_CC with two-digit row/column indices.",
      call. = FALSE
    )
  }

  compact <- gsub("_", "", barcodes, fixed = TRUE)
  coordinates <- data.frame(
    matrix_barcode = barcodes,
    array_row = as.integer(substr(compact, 1L, 2L)),
    array_col = as.integer(substr(compact, 3L, 4L)),
    stringsAsFactors = FALSE
  )
  coordinate_keys <- paste(coordinates$array_row, coordinates$array_col, sep = ":")
  if (anyDuplicated(coordinate_keys)) {
    stop("Matrix barcodes contain duplicate row/column coordinates.", call. = FALSE)
  }
  coordinates
}


#' Load DBiT spatial methylation matrices into a Seurat object
#'
#' Reads `mean_shrunken_residuals` and `methylation_fractions` 10x-style
#' directories into one Seurat v5 assay. The residual matrix is stored in the
#' `data` layer (analogous to `X` in the Python workflow), and methylation
#' fractions are stored in the `methylation` layer. Matrix barcodes such as
#' `02_34` are matched to sequence barcodes through `array_row` and
#' `array_col` in the tissue-position table.
#'
#' The full-resolution grayscale image is resized in memory and attached as a
#' `VisiumV2` image. No intermediate image file is written.
#'
#' @param data_path Directory containing `mean_shrunken_residuals/` and
#'   `methylation_fractions/`.
#' @param residual_dir Residual-matrix subdirectory name.
#' @param methylation_dir Methylation-fraction subdirectory name.
#' @param image_name Name of the `VisiumV2` image slot.
#' @param assay Name of the Seurat assay containing both matrix layers. The
#'   active assay represents the residual matrix.
#' @param spot_radius Spot radius in micrometres.
#' @param um_per_px_fullres Micrometres per full-resolution pixel.
#' @param fullres_w Full-resolution image width. Inferred from the PNG when NULL.
#' @param fullres_h Full-resolution image height. Inferred from the PNG when NULL.
#' @param image_scale ImageMagick resize geometry, for example `"20%"`.
#' @param verbose Print progress messages.
#' @return A Seurat object whose `data` layer contains mean shrunken residuals
#'   and whose `methylation` layer contains methylation fractions, plus spatial
#'   metadata, spatial/physical reductions, and a `VisiumV2` image.
LoadDBiTMethylation <- function(
  data_path,
  residual_dir = "mean_shrunken_residuals",
  methylation_dir = "methylation_fractions",
  image_name = "ssDNA",
  assay = "Residuals",
  spot_radius,
  um_per_px_fullres = 0.294,
  fullres_w = NULL,
  fullres_h = NULL,
  image_scale = "20%",
  verbose = TRUE
) {
  data_path <- normalizePath(data_path, mustWork = TRUE)
  residual_path <- file.path(data_path, residual_dir)
  methylation_path <- file.path(data_path, methylation_dir)
  position_path <- file.path(residual_path, "tissue_positions.tsv.gz")
  image_path <- file.path(residual_path, "tissue_raw_image.png")

  missing_directories <- c(residual_path, methylation_path)[
    !dir.exists(c(residual_path, methylation_path))
  ]
  if (length(missing_directories) > 0L) {
    stop(
      "Missing matrix directory/directories: ",
      paste(missing_directories, collapse = ", "),
      call. = FALSE
    )
  }
  if (!file.exists(position_path)) {
    stop("Missing tissue positions: ", position_path, call. = FALSE)
  }
  if (!file.exists(image_path)) {
    stop("Missing ssDNA image: ", image_path, call. = FALSE)
  }

  residuals <- .read_single_dbit_matrix(
    residual_path,
    matrix_name = "mean shrunken residuals",
    verbose = verbose
  )
  methylation <- .read_single_dbit_matrix(
    methylation_path,
    matrix_name = "methylation fractions",
    verbose = verbose
  )
  if (!identical(dim(residuals), dim(methylation)) ||
      !identical(rownames(residuals), rownames(methylation)) ||
      !identical(colnames(residuals), colnames(methylation))) {
    stop(
      "Residual and methylation matrices must have identical features and barcodes.",
      call. = FALSE
    )
  }

  matrix_coordinates <- .parse_dbit_matrix_barcodes(colnames(residuals))
  positions <- read.delim(position_path, check.names = FALSE)
  required_position_columns <- c(
    "barcode", "in_tissue", "array_row", "array_col",
    "pxl_row_in_fullres", "pxl_col_in_fullres"
  )
  missing_columns <- setdiff(required_position_columns, colnames(positions))
  if (length(missing_columns) > 0L) {
    stop(
      "Missing tissue-position column(s): ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  integer_columns <- required_position_columns[-1L]
  for (column in integer_columns) {
    values <- suppressWarnings(as.numeric(positions[[column]]))
    if (anyNA(values) || any(!is.finite(values)) || any(values != floor(values))) {
      stop("Position column '", column, "' must contain integers.", call. = FALSE)
    }
    positions[[column]] <- as.integer(values)
  }
  if (anyDuplicated(positions$barcode)) {
    stop("Tissue positions contain duplicate sequence barcodes.", call. = FALSE)
  }
  position_keys <- paste(positions$array_row, positions$array_col, sep = ":")
  if (anyDuplicated(position_keys)) {
    stop("Tissue positions contain duplicate row/column coordinates.", call. = FALSE)
  }

  matrix_keys <- paste(
    matrix_coordinates$array_row,
    matrix_coordinates$array_col,
    sep = ":"
  )
  position_index <- match(matrix_keys, position_keys)
  if (anyNA(position_index)) {
    missing_index <- which(is.na(position_index))
    preview_index <- head(missing_index, 5L)
    preview <- paste(
      paste0(
        matrix_coordinates$matrix_barcode[preview_index], " -> (",
        matrix_coordinates$array_row[preview_index], ", ",
        matrix_coordinates$array_col[preview_index], ")"
      ),
      collapse = ", "
    )
    stop(
      length(missing_index), " matrix barcode(s) have no tissue position: ",
      preview,
      if (length(missing_index) > 5L) " ..." else "",
      call. = FALSE
    )
  }

  matched_positions <- positions[position_index, , drop = FALSE]
  cell_barcodes <- matched_positions$barcode
  if (anyDuplicated(cell_barcodes)) {
    stop("Matched tissue positions contain duplicate barcodes.", call. = FALSE)
  }
  colnames(residuals) <- cell_barcodes
  colnames(methylation) <- cell_barcodes
  rownames(matched_positions) <- cell_barcodes
  residual_assay <- SeuratObject::CreateAssay5Object(data = residuals)
  SeuratObject::LayerData(
    residual_assay,
    layer = "methylation"
  ) <- methylation
  seu <- suppressWarnings(
    SeuratObject::CreateSeuratObject(
      counts = residual_assay,
      assay = assay
    )
  )
  seu$in_tissue <- matched_positions[SeuratObject::Cells(seu), "in_tissue"]
  seu$array_row <- matched_positions[SeuratObject::Cells(seu), "array_row"]
  seu$array_col <- matched_positions[SeuratObject::Cells(seu), "array_col"]
  seu@misc$matrix_sources <- list(
    data = residual_dir,
    methylation = methylation_dir
  )

  coords <- as.matrix(
    matched_positions[
      SeuratObject::Cells(seu),
      c("pxl_col_in_fullres", "pxl_row_in_fullres")
    ]
  )
  colnames(coords) <- c("coords_1", "coords_2")
  seu[["spatial"]] <- SeuratObject::CreateDimReducObject(
    embeddings = coords,
    key = "coords_",
    assay = assay
  )

  raw_dimensions <- read_png_dimensions(image_path)
  if (is.null(fullres_w)) fullres_w <- raw_dimensions[["width"]]
  if (is.null(fullres_h)) fullres_h <- raw_dimensions[["height"]]
  if (isTRUE(verbose)) {
    message("Reading and scaling ssDNA image ", image_path, " to ", image_scale)
  }
  image_object <- magick::image_read(image_path)
  image_object <- magick::image_scale(image_object, geometry = image_scale)
  ssDNA <- magick::image_data(image_object, channels = "gray")[1, , ]
  ssDNA <- t(ssDNA)
  storage.mode(ssDNA) <- "double"
  ssDNA <- ssDNA / 255
  rm(image_object)

  tissue_lowres_scalef <- dim(ssDNA)[2L] / as.numeric(fullres_w)
  if (isTRUE(verbose)) {
    message(sprintf(
      "tissue_lowres_scalef (w/h) = %.6f / %.6f",
      tissue_lowres_scalef,
      dim(ssDNA)[1L] / as.numeric(fullres_h)
    ))
  }
  spot_radius_px <- as.numeric(spot_radius) / um_per_px_fullres
  sf <- Seurat::scalefactors(
    spot = spot_radius_px,
    fiducial = 100,
    hires = tissue_lowres_scalef,
    lowres = tissue_lowres_scalef
  )
  coord_img <- data.frame(
    imagecol = coords[, 1L],
    imagerow = coords[, 2L],
    row.names = SeuratObject::Cells(seu)
  )
  fov <- SeuratObject::CreateFOV(
    coord_img,
    type = "centroids",
    radius = sf[["spot"]],
    assay = assay,
    key = paste0(image_name, "_")
  )
  seu[[image_name]] <- methods::new(
    Class = "VisiumV2",
    boundaries = fov@boundaries,
    molecules = fov@molecules,
    assay = fov@assay,
    key = fov@key,
    image = ssDNA,
    scale.factors = sf,
    coords_x_orientation = "horizontal",
    misc = list(um_per_px_fullres = um_per_px_fullres)
  )
  SeuratObject::DefaultAssay(seu) <- assay

  coords_um <- coords * um_per_px_fullres
  colnames(coords_um) <- c("physical_1", "physical_2")
  seu[["physical"]] <- SeuratObject::CreateDimReducObject(
    embeddings = coords_um,
    key = "physical_",
    assay = assay
  )

  if (isTRUE(verbose)) {
    message(sprintf(
      paste0(
        "Loaded %d spots x %d features; layers='data' (residuals), ",
        "'methylation'; image='%s'"
      ),
      ncol(seu), nrow(seu), image_name
    ))
  }
  seu
}
