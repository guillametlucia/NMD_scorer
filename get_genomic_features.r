library(dplyr)
library(biomaRt)

input_csv <- "inputfile"
df <- read.csv(input_csv)


df <- df %>% rename(ensembl_transcript_id = TranscriptID)
df$ensembl_transcript_id <- as.character(df$ensembl_transcript_id)
df$ensembl_transcript_id <- sub("\\..*", "", df$ensembl_transcript_id)

ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")

# Define a function to get the summarized data for each transcript
get_transcript_summary <- function(transcript_id) {
  result <- getBM(
    attributes = c('ensembl_transcript_id', 'cds_length', 'transcript_length', 
                   '5_utr_start', '5_utr_end', 
                   '3_utr_start', '3_utr_end', 'strand'),
    filters = 'ensembl_transcript_id', 
    values = transcript_id, 
    mart = ensembl
  )
    # Get the cDNA sequence
  cdna_sequence <- getSequence(id = transcript_id, 
                               type = "ensembl_transcript_id", 
                               seqType = "cdna", 
                               mart = ensembl)
  cdna_value <- ifelse(is.null(cdna_sequence) || nrow(cdna_sequence) == 0, NA, cdna_sequence$cdna[1])
  

  if (nrow(result) > 0) {
    # Summarize UTR start and end positions based on strand
    summarized_result <- result %>%
      summarise(
        cds_length = first(cds_length),
        transcript_length = first(transcript_length),
        strand = first(strand),
        utr5_start = ifelse(strand[1] == 1, min(`5_utr_start`, na.rm = TRUE), max(`3_utr_start`, na.rm = TRUE)),
        utr5_end = ifelse(strand[1] == 1, max(`5_utr_end`, na.rm = TRUE), min(`3_utr_end`, na.rm = TRUE)),
        utr3_start = ifelse(strand[1] == 1, min(`3_utr_start`, na.rm = TRUE), max(`5_utr_start`, na.rm = TRUE)),
        utr3_end = ifelse(strand[1] == 1, max(`3_utr_end`, na.rm = TRUE), min(`5_utr_end`, na.rm = TRUE))
      )
    summarized_result$cdna_sequence <- cdna_value
    

    
  } else {
    summarized_result <- data.frame(
                                    cds_length = NA,
                                    transcript_length = NA,
                                    utr5_start = NA,
                                    utr5_end = NA,
                                    utr3_start = NA,
                                    utr3_end = NA,
                                    strand = NA,
                                    cdna_sequence = cdna_value)
  }
  
  return(summarized_result)
}

# Apply the function to each row in the sampled DataFrame
df_info <- do.call(rbind, lapply(df$ensembl_transcript_id, function(id) {
  get_transcript_summary(id)
}))


# Merge the summarized results back into the sampled DataFrame
print(paste("Original df rows:", nrow(df)))
print(paste("Summarized df_info rows:", nrow(df_info)))
df <- cbind(df, df_info)
print(paste("Merged df rows:", nrow(df)))


output_csv <- "outputfile"
write.csv(df, output_csv, row.names = FALSE)
