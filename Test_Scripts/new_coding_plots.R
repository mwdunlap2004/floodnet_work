library('tidyverse')
library('jsonlite')
library('leaflet')
library('htmltools')
library('lubridate') # Added for date manipulation in new graphs

# --- Data Loading & Cleaning ---
script_path_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_path_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_path_arg[1]), mustWork = FALSE))
} else if (basename(getwd()) %in% c("Test_Scripts", "Finalized_Scripts")) {
  normalizePath(file.path(getwd()), mustWork = FALSE)
} else {
  getwd()
}

project_root <- if (basename(script_dir) %in% c("Test_Scripts", "Finalized_Scripts")) {
  normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
} else {
  normalizePath(script_dir, mustWork = FALSE)
}

resolve_input_path <- function(filename) {
  candidates <- c(
    file.path(getwd(), filename),
    file.path(project_root, filename),
    file.path(project_root, "Data_Files", filename),
    file.path(project_root, "Test_Scripts", filename),
    file.path(project_root, "Finalized_Scripts", filename)
  )
  for (p in candidates) {
    if (file.exists(p)) return(p)
  }
  stop(
    paste0(
      "Could not find input file: ", filename, "\nSearched:\n",
      paste(candidates, collapse = "\n")
    )
  )
}

PLOT_DIR <- file.path(project_root, "Images_or_plots")
if (!dir.exists(PLOT_DIR)) dir.create(PLOT_DIR, recursive = TRUE)

save_plot <- function(filename, width = 10, height = 6, dpi = 300) {
  out_path <- file.path(PLOT_DIR, filename)
  ggsave(filename = out_path, plot = last_plot(), width = width, height = height, dpi = dpi)
  message("Saved plot: ", out_path)
}

rain_data = read.csv(resolve_input_path('got_rain.csv'))
full_data=read.csv(resolve_input_path('full_data.csv'))
master_data <- read.csv(resolve_input_path('master_data.csv'))

full_data <- full_data |> mutate(borough = str_trim(borough)) 
rain_data <- rain_data |> mutate(borough = str_trim(borough)) 

# --- Visualizations ---

# SLIDE TITLE: Average Rainfall Accumulation by Borough
# DESCRIPTION: A high-level comparison showing which boroughs receive the highest average precipitation per recorded event.
rain_data |>
  group_by(borough) |>
  summarise(mean_precip = mean(depth_inches, na.rm = TRUE)) |>
  ggplot(aes(x = borough, y = mean_precip)) +
  geom_col() + theme_classic()
save_plot("avg_rainfall_by_borough_new.png")


# SLIDE TITLE: Average Flood Depth by Borough
# DESCRIPTION: Unlike rainfall (what falls from the sky), this chart shows the average standing water depth recorded by flood sensors, highlighting areas with drainage challenges.
full_data |>
  group_by(borough) |>
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE)) |>
  ggplot(aes(x = borough, y = mean_depth)) +
  geom_col() + theme_classic()
save_plot("avg_flood_depth_by_borough_new.png")


# SLIDE TITLE: Top 10 Most Severe Flood Events (City-Wide)
# DESCRIPTION: Identifies the specific dates with the highest average flood depths across the entire city, pinpointing our most extreme historical storms.
full_data |> group_by(date) |> 
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE)) |>
  slice_max(mean_depth, n = 10) |> 
  ggplot(aes(x = date, y = mean_depth)) +
  geom_col() + theme_classic()
save_plot("top10_citywide_flood_events_new.png")


# SLIDE TITLE: Raw Timeline of Sensor Readings
# DESCRIPTION: A scatter-line view of all raw data points. Note the density of lines indicating sensor uptime and the spikes representing storm events.
full_data |>
  ggplot(aes(x = date, y = depth_inches, color = borough)) +
  geom_line() +
  theme_classic()
save_plot("raw_sensor_timeline_new.png", width = 12, height = 7)


# SLIDE TITLE: Longitudinal Flood Trends by Borough
# DESCRIPTION: Aggregated daily averages reveal the synchronization of flood events across boroughs while highlighting local deviations.
full_data |>
  group_by(date, borough) |> 
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE)) |>
  ggplot(aes(x = date, y = mean_depth, color = borough)) +
  geom_line() +
  theme_classic()
save_plot("borough_longitudinal_flood_trends_new.png", width = 12, height = 7)


# SLIDE TITLE: Flood Trends (Date Corrected)
# DESCRIPTION: (Same as above, but ensures chronological accuracy on the X-axis for clearer trend analysis).
df_plot <- full_data |>
  group_by(date, borough) |>
  summarise(mean_depth = mean(depth_inches), .groups = "drop")
df_plot$date <- as.Date(df_plot$date)
ggplot(df_plot, aes(x = date, y = mean_depth, color = borough, group = borough)) +
  geom_line() + theme_classic()
save_plot("borough_flood_trends_date_corrected_new.png", width = 12, height = 7)


# --- Focused Event Analysis ---

# SLIDE TITLE: Data Preparation for Target Dates
# DESCRIPTION: Filtering the dataset to isolate specific high-interest storm dates for detailed impact analysis.
rainiest_days <- full_data |>
  group_by(date) |>
  summarise(mean_precip = mean(depth_inches, na.rm = TRUE)) |>
  arrange(desc(mean_precip)) |>
  slice_head(n = 10)

selected_dates <- as.Date(c(
  "2024-10-18", "2024-10-20", "2024-10-19", "2024-01-13",
  "2022-12-23", "2025-10-18", "2021-08-22", "2024-01-10",
  "2024-08-26", "2025-10-30"
))

rainiest_days <- full_data |> filter(date %in% selected_dates)


# SLIDE TITLE: Precipitation Levels During Key Storm Events
# DESCRIPTION: A comparison of the total city-wide impact for our selected list of major weather events.
plot_data <- rainiest_days |>
  group_by(date) |>
  summarise(mean_precip = mean(depth_inches, na.rm = TRUE), .groups = "drop")

ggplot(plot_data, aes(x = factor(date), y = mean_precip)) +
  geom_col(fill = "steelblue") +
  theme_classic() +
  labs(
    title = "Rainiest Selected Days",
    x = "Date",
    y = "Mean Precipitation (inches)"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot("selected_rainiest_days_citywide_new.png")


# SLIDE TITLE: Storm Impact Breakdown by Borough
# DESCRIPTION: Shows how the same storm systems affected boroughs differently, revealing geographic disparities in flooding severity.
plot_data <- rainiest_days |>
  group_by(date, borough) |>
  summarise(mean_precip = mean(depth_inches, na.rm = TRUE), .groups = "drop")

ggplot(plot_data, aes(x = factor(date), y = mean_precip, fill = borough)) +
  geom_col(position = "dodge") +
  theme_classic() +
  labs(title = "Rainiest Selected Days by Borough", x = "Date", y = "Mean Precipitation") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot("selected_rainiest_days_by_borough_new.png", width = 12, height = 7)


# --- Ranked Analysis ---

# SLIDE TITLE: The "Worst Day" for Each Borough
# DESCRIPTION: This chart uncouples the boroughs to show the top 10 unique flood events specific to each location, independent of city-wide averages.
full_data |>
  group_by(borough, date) |>
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE), .groups = "drop") |>
  group_by(borough) |>
  slice_max(mean_depth, n = 10) |>
  ggplot(aes(x = factor(date), y = mean_depth, fill = borough)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~borough, scales = "free_x") + 
  theme_classic() +
  labs(
    title = "Top 10 Rainiest Days Per Borough (Individually Ranked)",
    x = "Date",
    y = "Mean Depth (inches)"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot("top10_days_per_borough_ranked_new.png", width = 12, height = 8)


# SLIDE TITLE: Shared vs. Localized Storm Events
# DESCRIPTION: Dark blue bars indicate massive storms that hit multiple boroughs simultaneously. Light blue bars highlight localized flooding events unique to that specific borough.
ranked_days <- full_data |>
  group_by(borough, date) |>
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE), .groups = "drop") |>
  group_by(borough) |>
  slice_max(mean_depth, n = 10) |>
  ungroup()

date_counts <- ranked_days |> count(date, name = "borough_count")

plot_data <- ranked_days |>
  left_join(date_counts, by = "date") |>
  mutate(
    event_type = if_else(borough_count > 1, "Shared Storm Event", "Local/Unique Event"),
    date_label = factor(date)
  )

ggplot(plot_data, aes(x = date_label, y = mean_depth, fill = event_type)) +
  geom_col() +
  facet_wrap(~borough, scales = "free_x") + 
  scale_fill_manual(values = c("Shared Storm Event" = "#2c3e50", "Local/Unique Event" = "#3498db")) +
  theme_classic() +
  labs(
    title = "Top 10 Rainiest Days by Borough",
    subtitle = "Dark bars = Dates that were top rain events in multiple boroughs",
    x = "Date",
    y = "Mean Precipitation (inches)",
    fill = "Event Type"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )
save_plot("shared_vs_local_storm_events_new.png", width = 12, height = 8)


# --- Geospatial & Street Analysis ---

# SLIDE TITLE: Interactive Map of Peak Flood Events
# DESCRIPTION: An interactive visualization showing the single highest flood depth recorded by every individual sensor. Larger circles indicate deeper historical maximums.
daily_stats <- master_data |>
  group_by(deployment_id, name_x, latitude, longitude, deploy_type, deploy_date) |>
  summarise(
    daily_mean = mean(depth_inches, na.rm = TRUE),
    .groups = "drop"
  )

top_gage_events <- daily_stats |>
  group_by(deployment_id) |>
  slice_max(daily_mean, n = 1, with_ties = FALSE) |>
  ungroup() |>
  filter(daily_mean > 0)

pal <- colorNumeric(palette = "Blues", domain = top_gage_events$daily_mean)

flood_map <- leaflet(data = top_gage_events) |>
  addProviderTiles(providers$CartoDB.Positron) |> 
  addCircleMarkers(
    ~longitude, ~latitude,
    radius = ~sqrt(daily_mean) * 8, 
    color = ~pal(daily_mean),
    fillColor = ~pal(daily_mean),
    fillOpacity = 0.7,
    stroke = TRUE,
    weight = 1,
    popup = ~paste0(
      "<b>Sensor:</b> ", name_x, "<br>",
      "<b>Event Date:</b> ", deploy_date, "<br>",
      "<b>Daily Avg Depth:</b> ", round(daily_mean, 2), " inches<br>",
      "<b>Type:</b> ", deploy_type
    ),
    label = ~paste0(name_x, ": ", round(daily_mean, 2), "\"")
  ) |>
  addLegend(
    "bottomright",
    pal = pal,
    values = ~daily_mean,
    title = "Max Daily Avg (in)",
    opacity = 1
  )

htmlwidgets::saveWidget(
  flood_map,
  file = file.path(PLOT_DIR, "peak_flood_events_map_new.html"),
  selfcontained = TRUE
)
message("Saved map: ", file.path(PLOT_DIR, "peak_flood_events_map_new.html"))


# SLIDE TITLE: Top 20 Most Flooded Intersections
# DESCRIPTION: A ranking of specific intersections based on average depth, strictly during flood events (excluding dry days).
master_data <- master_data %>% rename(name = name_x)

street_ranking <- master_data |>
  filter(depth_inches > 0) |>
  group_by(name) |>
  summarise(avg_flood_depth = mean(depth_inches, na.rm = TRUE)) |>
  arrange(desc(avg_flood_depth)) |>
  slice_head(n = 20) 

ggplot(street_ranking, aes(x = avg_flood_depth, y = reorder(name, avg_flood_depth))) +
  geom_col(fill = "steelblue") +
  theme_classic() +
  labs(
    title = "Top 20 Streets with the Deepest Floods",
    subtitle = "Average recorded depth (inches) when flooding occurs",
    x = "Average Depth (inches)",
    y = NULL 
  ) +
  theme(
    axis.text.y = element_text(size = 10), 
    plot.title = element_text(face = "bold")
  )
save_plot("top20_most_flooded_intersections_new.png", width = 12, height = 8)


# SLIDE TITLE: Top 20 Flooded Streets (Aggregated)
# DESCRIPTION: By splitting intersection names, we identify which specific avenues or streets appear most frequently in flood records.
street_data <- master_data |>
  filter(depth_inches > 0) |>
  separate(name, into = c("borough_code", "intersection"), sep = " - ", extra = "merge", fill = "right") |>
  separate(intersection, into = c("street_1", "street_2"), sep = "/", extra = "merge", fill = "right") |>
  pivot_longer(
    cols = c(street_1, street_2), 
    names_to = "street_pos", 
    values_to = "street_name"
  ) |>
  filter(!is.na(street_name)) |>
  mutate(street_name = str_trim(street_name))

street_ranking <- street_data |>
  group_by(street_name, borough_code) |>
  summarise(
    avg_flood_depth = mean(depth_inches, na.rm = TRUE),
    total_flood_events = n(), 
    .groups = "drop"
  ) |>
  arrange(desc(avg_flood_depth)) |>
  slice_head(n = 20)

ggplot(street_ranking, aes(x = avg_flood_depth, y = reorder(street_name, avg_flood_depth), fill = borough_code)) +
  geom_col() +
  theme_classic() +
  labs(
    title = "Top 20 Rainiest Avenues & Streets",
    subtitle = "Aggregated by individual street name (splitting intersections)",
    x = "Average Flood Depth (inches)",
    y = NULL,
    fill = "Borough"
  ) +
  theme(
    axis.text.y = element_text(size = 10),
    legend.position = "bottom"
  )
save_plot("top20_rainiest_streets_aggregated_new.png", width = 12, height = 8)


# SLIDE TITLE: Chronic Flooding Analysis (All Days)
# DESCRIPTION: This ranking includes "dry days" (0 inches). Streets at the top of this list likely suffer from chronic issues like tidal flooding or poor drainage, as water is present more often than not.
street_data_all <- master_data |>
  pivot_longer(
    cols = c(street1, street2), 
    names_to = "street_pos", 
    values_to = "street_name"
  ) |>
  filter(!is.na(street_name) & street_name != "") |>
  mutate(street_name = str_trim(street_name))

street_ranking_all <- street_data_all |>
  group_by(street_name) |>
  summarise(
    avg_depth_overall = mean(depth_inches, na.rm = TRUE),
    total_records = n(),
    .groups = "drop"
  ) |>
  arrange(desc(avg_depth_overall)) |>
  slice_head(n = 20)

ggplot(street_ranking_all, aes(x = avg_depth_overall, y = reorder(street_name, avg_depth_overall))) +
  geom_col(fill = "darkblue") +
  geom_text(aes(label = paste0("n=", total_records)), 
            hjust = -0.1, size = 3) +
  theme_classic() +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) + 
  labs(
    title = "Top 20 Streets by Overall Average Depth",
    subtitle = "Includes dry days (0.0). High ranking = Chronic flooding or Tidal issues.",
    x = "Average Depth (inches)",
    y = NULL
  )
save_plot("top20_streets_chronic_flooding_all_days_new.png", width = 12, height = 8)
