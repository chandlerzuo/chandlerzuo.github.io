# census data metadata
library(rjson)
library(rvest)

ACS_URL_PREFIX = "https://api.census.gov/data/2018/acs/acs1/"

#' Get ACS variables related to this analysis.
get_all_variables <- function() {
  meta_data <-
    fromJSON(file = paste(ACS_URL_PREFIX, "variables.json", sep = ""))
  old_age_group_labels <- c()
  old_age_group_vars <- c()
  all_pop_var <- NULL
  for (var in names(meta_data$variables)) {
    if (grepl("B01001_", var)) {
      label <- meta_data$variables[[var]]$label
      if (grepl("!!Male!!6", label) |
          grepl("!!Male!!7", label) |
          grepl("!!Male!!8", label) | grepl("!!Female!!6", label) |
          grepl("!!Female!!7", label) |
          grepl("!!Female!!8", label)) {
        old_age_group_labels <-
          append(old_age_group_labels, meta_data$variables[[var]]$label)
        old_age_group_vars <- append(old_age_group_vars, var)
      } else if (label == "Estimate!!Total") {
        all_pop_var <- var
      }
    }
  }
  return(
    list(
      old_age_groups = data.frame(variable = old_age_group_vars,
                                  label = old_age_group_labels),
      pop_var = all_pop_var,
      median_household_income_var = "B19013_001E"
    )
  )
}

# Get county level data
get_county_level_data <- function(old_age_group_vars,
                                  pop_var,
                                  median_household_income_var) {
  api_url <-
    paste(ACS_URL_PREFIX,
          "?get=NAME,",
          paste(
            c(old_age_group_vars,
              pop_var,
              median_household_income_var),
            collapse = ","
          ),
          "&for=county:*",
          sep = "")
  
  census_data <- fromJSON(file = api_url)
  census_mat <- c()
  for (i in 2:length(census_data)) {
    census_mat <- rbind(census_mat, census_data[[i]])
  }
  census_table <-
    as.data.frame(census_mat, stringsAsFactors = FALSE)
  colnames(census_table) <- census_data[[1]]
  
  census_table[, "PROP_GE60"] = apply(census_table[, old_age_group_vars], 1, sum) / census_table[, pop_var]
  return(census_table)
}

# Get COVID County Data
get_covid_data <- function() {
  url <- "state_table.html"
  text <-
    read_html(url) %>%
    html_nodes(css = ".jsx-742282485 .row") %>%
    html_nodes(
      xpath = paste(
        '//*[contains(concat(" ",normalize-space(@class)," ")," ',
        'jsx-742282485 ")]/text()[last()]',
        sep = ''
      )
    ) %>%
    html_text()
  
  text <- gsub(",", "", text)
  .validate_line_partial <- function(line) {
    if (!suppressWarnings(is.na(as.integer(line[2]))) &
        !suppressWarnings(is.na(as.integer(line[3])))) {
      return(TRUE)
    }
    return(FALSE)
  }
  .validate_line <- function(line) {
    if (.validate_line_partial(line) & grepl("%", line[4]))
      return(TRUE)
    return(FALSE)
  }
  cur <- 7
  regions <- infected <- deaths <- c()
  while (cur < length(text)) {
    line <- text[cur:(cur + 3)]
    if (!.validate_line(line)) {
      cur <- cur + 1
      line <- text[cur:(cur + 3)]
      if (!.validate_line(line)) {
        if (.validate_line_partial(text[(cur - 1):(cur + 1)]) &
            .validate_line_partial(text[(cur + 2):(cur + 5)])) {
          regions <- c(regions, text[cur - 1])
          infected <- c(infected, as.integer(text[cur]))
          deaths <- c(deaths, as.integer(text[cur + 1]))
          cur <- cur + 2
          next
        } else {
          stop(paste("Check around", cur))
        }
      }
    }
    regions <- c(regions, line[1])
    infected <- c(infected, as.integer(line[2]))
    deaths <- c(deaths, as.integer(line[3]))
    cur <- cur + 4
  }
  dat <- data.frame(region = regions,
                    infected = infected,
                    death = deaths)
  return(dat)
}