#!/usr/bin/env Rscript
# Annotate MethSCAn VMR BED intervals against a matching local GTF.

CLUSTER_PALETTE <- c(
    "#E64B35", "#2F5597", "#4DBBD5", "#CC79A7", "#C5A3E0", "#D6A500", "#98DF8A"
)

parse_args <- function() {
    usage <- paste(
        "Usage: pixi run Rscript qc/annotate-vmr.R",
        "  --bed VMRs.bed --gtf annotation.gtf.gz --output DIR",
        "  [--promoter-upstream 3000] [--promoter-downstream 3000]",
        "",
        "BED coordinates must be 0-based, half-open.",
        "Only the first three BED columns are used for annotation.",
        "The output directory must not already exist.",
        sep = "\n"
    )
    values <- commandArgs(trailingOnly = TRUE)
    if (length(values) == 0L || any(values %in% c("--help", "-h"))) {
        cat(usage, "\n")
        quit(status = 0L)
    }
    if (length(values) %% 2L != 0L) stop(usage)
    keys <- sub("^--", "", values[seq.int(1L, length(values), 2L)])
    allowed <- c("bed", "gtf", "output", "promoter-upstream", "promoter-downstream")
    if (any(!keys %in% allowed) || anyDuplicated(keys)) stop(usage)
    args <- as.list(values[seq.int(2L, length(values), 2L)])
    names(args) <- keys
    if (!all(c("bed", "gtf", "output") %in% keys)) stop(usage)
    for (key in c("promoter-upstream", "promoter-downstream")) {
        value <- if (is.null(args[[key]])) 3000L else suppressWarnings(as.integer(args[[key]]))
        if (is.na(value) || value < 0L) stop("Invalid ", key)
        args[[key]] <- value
    }
    args
}

read_vmrs <- function(path) {
    frame <- read.delim(path, header = FALSE, comment.char = "#", stringsAsFactors = FALSE)
    if (ncol(frame) < 3L || nrow(frame) == 0L) stop("BED requires nonempty intervals.")
    names(frame)[1:3] <- c("chrom", "bed_start", "bed_end")
    coordinates <- c(frame$bed_start, frame$bed_end)
    if (!is.numeric(coordinates) || anyNA(frame) ||
        any(!is.finite(coordinates)) || any(coordinates != floor(coordinates)) ||
        any(frame$bed_start < 0) || any(frame$bed_end <= frame$bed_start)) {
        stop("Invalid BED coordinates or missing values.")
    }
    frame$vmr_id <- paste0(frame$chrom, ":", frame$bed_start, "-", frame$bed_end)
    if (anyDuplicated(frame$vmr_id)) stop("Duplicate VMR coordinates in BED.")
    frame$width_bp <- frame$bed_end - frame$bed_start
    frame
}

summarize_counts <- function(values, levels, denominator) {
    counts <- as.integer(table(factor(values, levels = levels)))
    data.frame(category = levels, count = counts, percentage = 100 * counts / denominator)
}

save_table <- function(frame, path) {
    write.table(frame, path, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
}

save_plot <- function(plot, stem, output, width = 5.4, height = 3.2) {
    ggplot2::ggsave(file.path(output, paste0(stem, ".png")), plot,
                   width = width, height = height, dpi = 300, bg = "white")
}

plot_annotation_pie <- function(summary) {
    category_levels <- summary$category
    plot_summary <- summary
    plot_summary$category <- factor(plot_summary$category, levels = category_levels)
    boundaries <- c(0, cumsum(plot_summary$count) / sum(plot_summary$count)) * 2 * pi
    plot_summary$angle <- head(boundaries, -1L) + diff(boundaries) / 2
    polygons <- do.call(rbind, lapply(seq_len(nrow(plot_summary)), function(index) {
        angle <- seq(boundaries[index], boundaries[index + 1L], length.out = 400)
        data.frame(x = c(0, sin(angle), 0), y = c(0, cos(angle), 0),
                   category = category_levels[index])
    }))
    polygons$category <- factor(polygons$category, levels = category_levels)
    colors <- setNames(CLUSTER_PALETTE[seq_along(category_levels)], category_levels)
    legend_labels <- sprintf("%s (%.2f%%)", category_levels, plot_summary$percentage)
    plot <- ggplot2::ggplot(polygons, ggplot2::aes(x = x, y = y, group = category, fill = category)) +
        ggplot2::geom_polygon(color = "white", linewidth = 0.05) +
        ggplot2::geom_text(
            data = plot_summary[plot_summary$percentage >= 3, , drop = FALSE],
            ggplot2::aes(x = 0.55 * sin(angle), y = 0.55 * cos(angle),
                         label = sprintf("%.2f%%", percentage)),
            inherit.aes = FALSE, color = "#111111", size = 9 / ggplot2::.pt, fontface = "bold"
        ) +
        ggplot2::coord_equal(xlim = c(-1.04, 1.04), ylim = c(-1.24, 1.04),
                             expand = FALSE, clip = "off") +
        ggplot2::scale_fill_manual(values = colors, breaks = category_levels,
                                  labels = legend_labels, drop = FALSE) +
        ggplot2::labs(fill = NULL) +
        ggplot2::theme_void(base_size = 10) +
        ggplot2::theme(legend.text = ggplot2::element_text(size = 9),
                       legend.justification = "top",
                       legend.key.height = grid::unit(0.45, "cm"),
                       legend.key.width = grid::unit(0.4, "cm"),
                       plot.margin = ggplot2::margin(4, 4, 4, 4))
    positive <- which(plot_summary$count > 0)
    smallest <- positive[which.min(plot_summary$count[positive])]
    if (plot_summary$percentage[smallest] >= 1) return(plot)
    rare <- plot_summary[smallest, , drop = FALSE]
    plot <- plot +
        ggplot2::geom_segment(data = rare,
            ggplot2::aes(x = 0.98 * sin(angle), xend = 1.10 * sin(angle),
                         y = 0.98 * cos(angle), yend = 1.10 * cos(angle)),
            inherit.aes = FALSE, color = "#444444", linewidth = 0.3) +
        ggplot2::geom_text(data = rare,
            ggplot2::aes(x = 1.18 * sin(angle), y = 1.18 * cos(angle),
                         label = sprintf("%.2f%%", percentage)),
            inherit.aes = FALSE, color = "#222222", size = 9 / ggplot2::.pt)

    # Crop the true pie geometry around the rare slice; do not enlarge its angle.
    neighbors <- seq.int(max(1L, smallest - 1L), min(nrow(plot_summary), smallest + 1L))
    zoom_polygons <- polygons[polygons$category %in% category_levels[neighbors], , drop = FALSE]
    middle <- mean(boundaries[c(smallest, smallest + 1L)])
    center <- 0.82 * c(sin(middle), cos(middle))
    half_width <- 0.06
    zoom <- ggplot2::ggplot(zoom_polygons,
        ggplot2::aes(x = x, y = y, group = category, fill = category)) +
        ggplot2::geom_polygon(color = NA) +
        ggplot2::scale_fill_manual(values = colors, guide = "none") +
        ggplot2::coord_equal(xlim = center[1] + c(-half_width, half_width),
                            ylim = center[2] + c(-half_width, half_width), expand = FALSE) +
        ggplot2::labs(caption = sprintf("Local zoom: %s VMRs", rare$count)) +
        ggplot2::theme_void(base_size = 8) +
        ggplot2::theme(plot.caption = ggplot2::element_text(size = 8, hjust = 0),
                       panel.border = ggplot2::element_rect(color = "#666666", fill = NA, linewidth = 0.4),
                       plot.background = ggplot2::element_rect(fill = "white", color = NA),
                       plot.margin = ggplot2::margin(2, 2, 2, 2))
    plot + patchwork::inset_element(zoom, left = 0.68, bottom = 0.04,
                                    right = 0.86, top = 0.42, align_to = "full")
}

plot_chromosome_distribution <- function(summary) {
    plot_summary <- summary
    plot_summary$chromosome <- factor(plot_summary$chromosome, levels = summary$chromosome)
    ggplot2::ggplot(plot_summary, ggplot2::aes(x = chromosome, y = count)) +
        ggplot2::geom_col(fill = "#4DBBD5", width = 0.75) +
        ggplot2::labs(x = NULL, y = "Number of VMRs") +
        ggplot2::theme_classic(base_size = 10) +
        ggplot2::theme(axis.title.y = ggplot2::element_text(size = 10),
                       axis.text = ggplot2::element_text(size = 9, color = "#222222"),
                       axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
                       plot.margin = ggplot2::margin(4, 4, 4, 4))
}

main <- function() {
    args <- parse_args()
    suppressPackageStartupMessages({
        library(ChIPseeker)
        library(GenomicRanges)
        library(ggplot2)
    })
    bed_path <- normalizePath(args$bed, mustWork = TRUE)
    gtf_path <- normalizePath(args$gtf, mustWork = TRUE)
    output <- path.expand(args$output)
    if (file.exists(output)) stop("Output already exists: ", output)
    vmrs <- read_vmrs(bed_path)
    message("Reading GTF and constructing TxDb: ", gtf_path)
    gtf <- rtracklayer::import(gtf_path, format = "gtf")
    if (!all(c("gene_id", "transcript_id", "type") %in% names(mcols(gtf)))) {
        stop("GTF lacks gene/transcript feature fields.")
    }
    absent <- setdiff(unique(vmrs$chrom), as.character(seqnames(gtf)))
    if (length(absent)) stop("VMR chromosomes absent from GTF: ", paste(absent, collapse = ", "))
    txdb <- txdbmaker::makeTxDbFromGRanges(gtf)
    rm(gtf)
    invisible(gc())
    peaks <- GRanges(seqnames = vmrs$chrom,
                     ranges = IRanges(start = vmrs$bed_start + 1, end = vmrs$bed_end),
                     strand = "*")
    mcols(peaks)$vmr_id <- vmrs$vmr_id
    priority <- c("Promoter", "5UTR", "3UTR", "Exon", "Intron", "Downstream", "Intergenic")
    message("Annotating ", nrow(vmrs), " VMRs")
    peak_anno <- annotatePeak(
        peaks, TxDb = txdb, level = "transcript", annoDb = NULL,
        tssRegion = c(-args[["promoter-upstream"]], args[["promoter-downstream"]]),
        genomicAnnotationPriority = priority
    )
    annotations <- as.data.frame(peak_anno)
    if (nrow(annotations) != nrow(vmrs) ||
        !setequal(annotations$vmr_id, vmrs$vmr_id)) stop("Annotation lost VMRs.")
    annotations <- annotations[match(vmrs$vmr_id, annotations$vmr_id), , drop = FALSE]
    if (anyNA(annotations$annotation) || anyNA(annotations$distanceToTSS)) {
        stop("Missing genomic annotations or TSS distances.")
    }
    annotations$category <- sub(" \\(.*$", "", annotations$annotation)
    category_levels <- c("Promoter", "5' UTR", "3' UTR", "Exon", "Intron", "Downstream", "Distal Intergenic")
    unexpected <- setdiff(unique(annotations$category), category_levels)
    if (length(unexpected)) stop("Unexpected categories: ", paste(unexpected, collapse = ", "))
    summary <- summarize_counts(annotations$category, category_levels, nrow(vmrs))
    chromosome_levels <- GenomeInfoDb::sortSeqlevels(unique(vmrs$chrom))
    chromosome_summary <- summarize_counts(vmrs$chrom, chromosome_levels, nrow(vmrs))
    names(chromosome_summary)[1] <- "chromosome"
    stopifnot(sum(summary$count) == nrow(vmrs), sum(chromosome_summary$count) == nrow(vmrs))
    dir.create(output, recursive = TRUE)
    save_table(summary, file.path(output, "annotation_summary.tsv"))
    save_table(chromosome_summary, file.path(output, "chromosome_summary.tsv"))
    save_plot(plot_annotation_pie(summary), "genomic_annotation", output)
    save_plot(plot_chromosome_distribution(chromosome_summary),
              "chromosome_distribution", output, width = 8, height = 3)
    print(summary, row.names = FALSE)
    message("Wrote analysis to ", normalizePath(output))
}

if (sys.nframe() == 0L) main()
