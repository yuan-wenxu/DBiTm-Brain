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
