rm(list = ls())

fig_dir = "D:/Dropbox/Public/blog/covid19/"
setwd(fig_dir)
source_dir = "D:/Documents/Github/chandlerzuo.github.io/codes/covid19"

library(stringr)
library(ggplot2)
library(tidyverse)
library(urbnmapr)

source(file.path(source_dir, "acs_data.R"))
source(file.path(source_dir, "covid_data.R"))

get_model_data <- function(use_cache = TRUE) {
  covid_data <- get_covid_data_from_csv()
  census_data <- get_census_data(use_cache = use_cache)
  census_data$state <-
    sapply(census_data$NAME, function(x)
      strsplit(x, ", ")[[1]][2])
  message("Census data have ", dim(census_data)[1], " rows.")
  census_data$county_name <-
    sapply(census_data$NAME, function(x)
      strsplit(x, ", ")[[1]][1])
  census_data$county <-
    sapply(census_data$county_name, function(x) {
      for (suffix in c("Municipality", "Borough", "Parish", "Municipio", "County")) {
        if (endsWith(x, suffix)) {
          x <- str_trim(strsplit(x, suffix)[[1]][1])
        }
      }
      return(x)
    })
  census_data$county <- sapply(census_data$NAME, function(x) {
    county <- strsplit(x, ", ")[[1]][1]
    for (suffix in c("Municipality", "Borough", "Parish", "Municipio", "County")) {
      if (endsWith(county, suffix)) {
        county <- str_trim(strsplit(county, suffix)[[1]][1])
      }
    }
    return(county)
  })
  merged_data <-
    merge(
      census_data,
      covid_data,
      all = TRUE,
      by.x = c("state", "county"),
      by.y = c("state", "region")
    )
  return(merged_data)
}

model_data <- get_model_data(use_cache = TRUE)
model_data$death_rate <- model_data$death / model_data$infected

jpeg(
  filename = file.path(fig_dir, "med_house_income.jpg"),
  width = 1200,
  height = 800
)
plot_data = model_data
ggplot(data = plot_data, aes(x = B19013_001E, y = death_rate)) + 
  geom_point(size=plot_data$infected / 10000, fill=NA, color="red") + 
  labs(x ="Median Household Income", y = "Death Rate") + 
  scale_y_log10() + geom_smooth(aes(weight=plot_data$infected)) + 
  theme(
    axis.title.x = element_text(size=20),
    axis.title.y = element_text(size=20),
    axis.text.x = element_text(size=18),
    axis.text.y = element_text(size=18)
  )
dev.off()

jpeg(
  filename = file.path(fig_dir, "prop_ge60.jpg"),
  width = 1200,
  height = 800
)
plot_data = model_data
ggplot(data = plot_data, aes(x = PROP_GE60, y = death_rate)) + 
  geom_point(size=plot_data$infected / 10000, fill=NA, color="red") + 
  labs(x ="Proportion of Population Older than 60", y = "Death Rate") + 
  scale_y_log10() + geom_smooth(aes(weight=plot_data$infected)) + 
  theme(
    axis.title.x = element_text(size=20),
    axis.title.y = element_text(size=20),
    axis.text.x = element_text(size=18),
    axis.text.y = element_text(size=18)
  )
dev.off()


jpeg(
  filename = file.path(fig_dir, "prop_female.jpg"),
  width = 1200,
  height = 800
)
plot_data = model_data
ggplot(data = plot_data, aes(x = PROP_FEMALE, y = death_rate)) + 
  geom_point(size=plot_data$infected / 10000, fill=NA, color="red") + 
  labs(x ="Proportion of Female", y = "Death Rate") + 
  scale_y_log10() + geom_smooth(aes(weight=plot_data$infected)) + 
  theme(
    axis.title.x = element_text(size=20),
    axis.title.y = element_text(size=20),
    axis.text.x = element_text(size=18),
    axis.text.y = element_text(size=18)
  ) + xlim(c(0.45, 0.55))
dev.off()


jpeg(
  filename = file.path(fig_dir, "insurance.jpg"),
  width = 1200,
  height = 800
)
plot_data = model_data
ggplot(data = plot_data, aes(x = PROP_HEALTH_INS, y = death_rate)) + 
  geom_point(size=plot_data$infected / 10000, fill=NA, color="red") + 
  labs(x ="Proportion of Population with Health Insurance", y = "Death Rate") + 
  scale_y_log10() + geom_smooth(aes(weight=plot_data$infected)) + 
  theme(
    axis.title.x = element_text(size=20),
    axis.title.y = element_text(size=20),
    axis.text.x = element_text(size=18),
    axis.text.y = element_text(size=18)
  )
dev.off()


jpeg(
  filename = file.path(fig_dir, "asian.jpg"),
  width = 1200,
  height = 800
)
plot_data = model_data
ggplot(data = plot_data, aes(x = PROP_ASIAN_ALONE, y = death_rate)) + 
  geom_point(size=plot_data$infected / 10000, fill=NA, color="red") + 
  labs(x ="Proportion of Asian Alone Families", y = "Death Rate") + 
  scale_y_log10() + geom_smooth(aes(weight=plot_data$infected)) + 
  theme(
    axis.title.x = element_text(size=20),
    axis.title.y = element_text(size=20),
    axis.text.x = element_text(size=18),
    axis.text.y = element_text(size=18)
  )
dev.off()



summary(model <-
          glm(
            cbind(death, infected) ~ B19013_001E,
            data = model_data[-c(1985, 1986, 2016, 2007, 3239), ],
            family = "binomial"
          ))

summary(model <-
          glm(
            cbind(death, infected) ~ PROP_GE60,
            data = model_data[-c(1985, 1986, 2016, 2007, 3239), ],
            family = "binomial"
          ))


summary(
  model <-
    glm(
      cbind(death, infected) ~
        PROP_GE60 + B19013_001E + PROP_HEALTH_INS,
      data = model_data[-c(1985, 1986, 2007, 2016, 3239), ],
      family = "binomial"
    )
)

summary(
  model <-
    glm(
      cbind(death, infected) ~
        B19013_001E + PROP_HEALTH_INS,
      data = model_data[-c(1985, 1986, 2007, 2016, 3239), ],
      family = "binomial"
    )
)


all_model_vars <-
  c(names(model_data)[grep("PROP_", names(model_data))], "B19013_001E")
all_model_vars <-
  setdiff(
    all_model_vars,
    c(
      "PROP_WHITE_ALONE",
      "PROP_WHITE_ALONE,_NOT_HISPANIC_OR_LATINO"
    )
  )

summary(model <-
          glm(cbind(death, infected) ~ .,
              data = model_data[-c(432, 1985, 1986, 3239, 2016, 2007), c(all_model_vars, "death", "infected")],
              family = "binomial"))

summary(model <-
          glm(cbind(death, infected) ~ .,
              data = model_data[-c(432, 1985, 1986, 3239, 2016, 2007),
                                setdiff(
                                  c(all_model_vars, "death", "infected"),
                                  c(
                                    "PROP_AMERICAN_INDIAN_AND_ALASKA_NATIVE_ALONE",
                                    "PROP_FEMALE",
                                    "PROP_HISPANIC_OR_LATINO",
                                    "PROP_HEALTH_INS",
                                    "B19013_001E"
                                  )
                                )],
              family = "binomial"))



model_data$predicted <-
  predict(model, newdata = model_data, type = "response")

model_data[c(432, 1985, 1986, 2007, 2016, 3239), c("state",
                                                   "county",
                                                   "infected",
                                                   "death")]

model_data$normalized <-
  log((1e-3 + model_data$death_rate) / model_data$predicted)
model_data$normalized[model_data$death_rate == 0] <- 0


map_data <-
  merge(
    model_data,
    counties,
    by.x = c("state", "county_name"),
    by.y = c("state_name", "county_name")
  )

jpeg(
  file = file.path(fig_dir, "normalized_death_rate.jpg"),
  width = 2000,
  height = 1200
)
map_data %>%
  ggplot(aes(long, lat, group = group, fill = normalized)) +
  geom_polygon(color = NA) +
  coord_map(projection = "albers",
            lat0 = 39,
            lat1 = 45) +
  labs(fill = "LOG(Actual Death Rate / Predicted Death Rate)", cex = 20) +
  scale_fill_gradient2(
    low = "#003300",
    high = "red",
    mid = "white",
    midpoint = 0
  ) + theme(
    legend.title=element_text(size=20),
    legend.text=element_text(size=20),
    axis.text.y=element_blank(),
    axis.text.x=element_blank(),
    axis.title.x=element_blank(),
    axis.title.y=element_blank(),
    legend.position="bottom"
  )
dev.off()


plot_data <-
  model_data[!is.na(model_data$infected) &
               !is.na(model_data$predicted) & model_data$infected > 1000,]
plot_data$rank <- rank(plot_data$predicted)
stacked_data <- rbind(data.frame(
  death_rate = c(plot_data$predicted, plot_data$death_rate),
  rank = rep(plot_data$rank, 2),
  infected=rep(plot_data$infected, 2),
  type = rep(c("Predicted", "Actual"), each = dim(plot_data)[1]),
  region = rep(plot_data$NAME, 2),
  stringsAsFactors = FALSE
))


p <- ggplot(
  data = stacked_data,
  aes(
    x = death_rate,
    y = rank,
    color = type,
    size=log(stacked_data$infected)
  ))  + 
  scale_y_continuous(breaks = 1:dim(plot_data)[1], labels = plot_data$NAME[order(plot_data$rank)]) + 
  ylab("") + xlab("Death Rate") +
  guides(color = guide_legend(title = "Death Rate"),
         size=guide_legend(title= "Log(Deaths)", fill="gray", color="gray")) + 
  theme(
    legend.title=element_text(size=20),
    legend.text=element_text(size=20),
    axis.text.y=element_text(size=18),
    axis.text.x=element_text(size=18),
    axis.title.x=element_text(size=18)
  ) +     geom_point()

for (i in seq(dim(plot_data)[1])) {
  xs <- c(plot_data$predicted[i], plot_data$death_rate[i])
  xmin <- min(min(xs) + 0.001, mean(xs))
  xmax <- max(max(xs) - 0.001, mean(xs))
  p <-
    p + geom_segment(
      x = xmin,
      xend = xmax,
      y = plot_data$rank[i],
      yend = plot_data$rank[i],
      color="gray",
      size=log(plot_data$death[i]) / 5,
      alpha=0.2
    )
}
jpeg(
  file = file.path(fig_dir, "actual_vs_predicted.jpg"),
  width = 1200,
  height = 1200
)
print(p)
dev.off()
  
