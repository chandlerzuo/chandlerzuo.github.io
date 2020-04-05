library(rjson)
library(rvest)

ACS_URL_PREFIX = "https://api.census.gov/data/2018/acs/"

#' Get ACS variables related to this analysis.
get_all_variables <- function() {
  meta_data <-
    fromJSON(file = paste(ACS_URL_PREFIX, "acs1/variables.json", sep = ""))
  old_age_group_vars <- c()
  female_pop_var <- NULL
  health_insurance_vars <- c()
  race_vars <- race_labels <- c()
  high_education_vars <- c()
  for (var in names(meta_data$variables)) {
    # B01001: Population by Sex and Age
    if (grepl("B01001_", var)) {
      label <- meta_data$variables[[var]]$label
      if (grepl("!!Male!!6", label) |
          grepl("!!Male!!7", label) |
          grepl("!!Male!!8", label) | grepl("!!Female!!6", label) |
          grepl("!!Female!!7", label) |
          grepl("!!Female!!8", label)) {
        old_age_group_vars <- append(old_age_group_vars, var)
      }
      if (label == "Estimate!!Total!!Female") {
        female_pop_var <- var
      }
      if (label == "Estimate!!Total!!Male") {
        male_pop_var <- var
      }
    }
    # C27001: Health Insurance Coverage by Sex by Age
    if (grepl("C27001_", var)) {
      label <- meta_data$variables[[var]]$label
      if (grepl("With health insurance coverage", label)) {
        health_insurance_vars <- append(health_insurance_vars, sub("C", "B", var))
      }
    }
    # B01001[A-Z]: Population for Races
    if (grepl("B01001[A-Z]_001E", var)) {
      race_vars <- append(race_vars, var)
      label <-
        strsplit(strsplit(meta_data$variables[[var]]$concept, "\\(")[[1]][2], "\\)")[[1]][1]
      race_labels <- append(race_labels, label)
    }
    # B15001 : SEX BY AGE BY EDUCATIONAL ATTAINMENT
    if (grepl("B15001_", var)) {
      label <- meta_data$variables[[var]]$label
      if (grepl("Graduate or professional degree", label) |
          grepl("Bachelor's degree", label)) {
        high_education_vars <- append(high_education_vars, var)
      }
    }
  }
  ret <-
    list(
      old_age_group_vars = old_age_group_vars,
      female_pop_var = female_pop_var,
      race_info = data.frame(
        variable = race_vars,
        label = race_labels,
        stringsAsFactors = FALSE
      ),
      pop_tot_var = "B01001_001E",
      health_insurance_vars = health_insurance_vars,
      health_insurance_tot_var = "B27001_001E",
      high_education_vars = high_education_vars,
      high_education_tot_var = "B15001_001E",
      median_household_income_var = "B19013_001E"
    )
  all_vars <- ret$race_info$variable
  for (var_info in ret) {
    if (class(var_info) == "character") {
      all_vars <- append(all_vars, var_info)
    }
  }
  ret$all_vars <- all_vars
  return(ret)
}

# Get county level data
get_census_data <- function(use_cache = TRUE) {
  CENSUS_DATA_CACHE <- "census.Rda"
  if (file.exists(CENSUS_DATA_CACHE) & use_cache) {
    load(CENSUS_DATA_CACHE)
    return(census_df)
  }
  meta <- get_all_variables()
  census_df <- NULL
  for (batch_id in seq(1, length(meta$all_vars), batch_size <-
                       10)) {
    api_url <-
      paste(
        ACS_URL_PREFIX,
        "acs5/?get=NAME,",
        paste(meta$all_vars[batch_id:min(batch_id + batch_size - 1, length(meta$all_vars))],
              collapse = ","),
        "&for=county:*",
        sep = ""
      )
    message("Get data from ", api_url)
    census_data <- fromJSON(file = api_url)
    # convert JSON to dataframe
    census_mat <- c()
    for (i in 2:length(census_data)) {
      if (is.list(census_data[[i]])) {
        census_data[[i]] <- as.character(census_data[[i]])
      }
      census_mat <- rbind(census_mat, census_data[[i]])
    }
    census_table <-
      as.data.frame(census_mat, stringsAsFactors = FALSE)
    census_table[census_table == "NULL"] <- NA
    colnames(census_table) <- census_data[[1]]
    # merge results from individual batches
    if (is.null(census_df)) {
      census_df <- census_table
    } else {
      census_df <- merge(census_df, census_table)
    }
  }
  
  census_df[, meta$all_vars] <-
    apply(census_df[, meta$all_vars], 2, as.integer)
  
  for (agg_info in list(
    list("PROP_GE60", meta$old_age_group_vars, meta$pop_tot_var),
    list("PROP_FEMALE", meta$female_pop_var, meta$pop_tot_var),
    list(
      "PROP_HIGH_EDU",
      meta$high_education_vars,
      meta$high_education_tot_var
    ),
    list(
      "PROP_HEALTH_INS",
      meta$health_insurance_vars,
      meta$health_insurance_tot_var
    )
  )) {
    if (length(agg_info[[2]]) > 1) {
      census_df[, agg_info[[1]]] <-
        apply(census_df[, agg_info[[2]]], 1, sum) / census_df[, agg_info[[3]]]
    } else {
      census_df[, agg_info[[1]]] <-
        census_df[, agg_info[[2]]] / census_df[, agg_info[[3]]]
    }
  }
  for (race_id in seq_len(dim(meta$race_info)[1])) {
    census_df[, paste("PROP", gsub(" ", "_", meta$race_info$label[race_id]), sep = "_")] <-
      census_df[, meta$race_info$variable[race_id]] / census_df[, meta$pop_tot_var]
  }
  save(census_df, file = CENSUS_DATA_CACHE)
  return(census_df)
}