library(rvest)

# Get COVID County Data
get_covid_data_from_html <- function(use_cache = TRUE) {
  COVID_DATA_CACHE <- "covid.csv"
  if (file.exists(COVID_DATA_CACHE) & use_cache) {
    load(COVID_DATA_CACHE)
    return(dat)
  }
  # url is snapshot at 12pm ET 4/4/2020
  url <- "state_table.html"
  text <-
    read_html(url) %>%
    html_nodes(css = ".jsx-1703765630 .row") %>%
    html_nodes(
      xpath = paste(
        '//*[contains(concat(" ",normalize-space(@class)," ")," ',
        'jsx-1703765630 ")]/text()[last()]',
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
  dat <- data.frame(
    region = regions,
    infected = infected,
    death = deaths,
    stringsAsFactors = FALSE
  )
  save(dat, file = COVID_DATA_CACHE)
  return(dat)
}

get_covid_data_from_csv <- function() {
  return(read.csv(
    "covid.csv",
    stringsAsFactors = FALSE,
    header = TRUE
  ))
}


assign_state <- function(dat, start_row = 1) {
  dat$state <- NA
  rowid = start_row
  i <- start_row + 1
  while (i <= nrow(dat)) {
    if (sum(dat$infected[(rowid + 1):i]) > dat$infected[rowid] &
        sum(dat$death[(rowid + 1):i]) > dat$death[rowid]) {
      dat$state[(rowid + 1):(i - 1)] <- dat$region[rowid]
      # message("Assign rows ", rowid + 1, " to ", i - 1, " to state ", dat$region[rowid])
      rowid <- i
    }
    i <- i + 1
  }
  return(dat)
}
