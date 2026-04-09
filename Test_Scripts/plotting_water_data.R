library('tidyverse')
library('jsonlite')
library('leaflet')
library('htmltools')

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
full_data <- full_data |>
  mutate(borough = str_trim(borough)) 
rain_data <- rain_data |> 
  mutate(borough = str_trim(borough)) 

master_data <- read.csv(resolve_input_path('master_data.csv'))


rain_data |>
  group_by(borough) |>
  summarise(mean_precip = mean(depth_inches, na.rm = TRUE)) |>
  ggplot(aes(x = borough, y = mean_precip)) +
  geom_col() +theme_classic()
save_plot("avg_rainfall_by_borough.png")



full_data |>
  group_by(borough) |>
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE)) |>
  ggplot(aes(x = borough, y = mean_depth)) +
  geom_col() +theme_classic()
save_plot("avg_flood_depth_by_borough.png")


full_data |> group_by(date) |> 
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE)) |>
  slice_max(mean_depth, n = 10) |> 
  ggplot(aes(x = date, y = mean_depth)) +
  geom_col() +theme_classic()
save_plot("top10_citywide_flood_events.png")

full_data |>
  ggplot(aes(x = date, y = depth_inches, color = borough)) +
  geom_line() +
  theme_classic()
save_plot("raw_sensor_timeline.png", width = 12, height = 7)

full_data |>
  group_by(date, borough) |> 
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE)) |>
  ggplot(aes(x = date, y = mean_depth, color = borough)) +
  geom_line() +
  theme_classic()
save_plot("borough_longitudinal_flood_trends.png", width = 12, height = 7)

df_plot <- full_data |>
  group_by(date, borough) |>
  summarise(mean_depth = mean(depth_inches), .groups = "drop")
df_plot$date <- as.Date(df_plot$date)
ggplot(df_plot, aes(x = date, y = mean_depth, color = borough, group = borough)) +
  geom_line()+theme_classic()
save_plot("borough_flood_trends_date_corrected.png", width = 12, height = 7)



rainiest_days <- full_data |>
  group_by(date) |>
  summarise(mean_precip = mean(depth_inches, na.rm = TRUE)) |>
  arrange(desc(mean_precip)) |>
  slice_head(n = 10)


selected_dates <- as.Date(c(
  "2024-10-18",
  "2024-10-20",
  "2024-10-19",
  "2024-01-13",
  "2022-12-23",
  "2025-10-18",
  "2021-08-22",
  "2024-01-10",
  "2024-08-26",
  "2025-10-30"
))

# Filter full_data to just these dates
rainiest_days <- full_data |>
  filter(date %in% selected_dates)

# Optional: check
rainiest_days |> arrange(date)


plot_data <- rainiest_days |>
  group_by(date) |>
  summarise(mean_precip = mean(depth_inches, na.rm = TRUE), .groups = "drop")

# Plot
ggplot(plot_data, aes(x = factor(date), y = mean_precip)) +
  geom_col(fill = "steelblue") +
  theme_classic() +
  labs(
    title = "Rainiest Selected Days",
    x = "Date",
    y = "Mean Precipitation (inches)"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot("selected_rainiest_days_citywide.png")

plot_data <- rainiest_days |>
  group_by(date, borough) |>
  summarise(mean_precip = mean(depth_inches, na.rm = TRUE), .groups = "drop")

ggplot(plot_data, aes(x = factor(date), y = mean_precip, fill = borough)) +
  geom_col(position = "dodge") +
  theme_classic() +
  labs(title = "Rainiest Selected Days by Borough", x = "Date", y = "Mean Precipitation") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot("selected_rainiest_days_by_borough.png", width = 12, height = 7)


full_data |> 
  count(borough, sort = TRUE)

# Clean borough column
full_data <- full_data |>
  mutate(borough = str_trim(borough))   # removes leading/trailing spaces

# Re-count
full_data |> 
  count(borough, sort = TRUE)



full_data |>
  # 1. Calculate daily average for every borough-date combo
  group_by(borough, date) |>
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE), .groups = "drop") |>
  
  # 2. Group by borough again to isolate the ranking
  group_by(borough) |>
  slice_max(mean_depth, n = 10) |>
  
  # 3. Plot
  ggplot(aes(x = factor(date), y = mean_depth, fill = borough)) +
  geom_col(show.legend = FALSE) +
  
  # scales = "free_x" allows each borough to have its own independent set of dates on the axis
  facet_wrap(~borough, scales = "free_x") + 
  
  theme_classic() +
  labs(
    title = "Top 10 Rainiest Days Per Borough (Individually Ranked)",
    x = "Date",
    y = "Mean Depth (inches)"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot("top10_days_per_borough_ranked.png", width = 12, height = 8)



# 1. Create the ranked list per borough
ranked_days <- full_data |>
  group_by(borough, date) |>
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE), .groups = "drop") |>
  group_by(borough) |>
  slice_max(mean_depth, n = 10) |>
  ungroup()

# 2. Count how many boroughs share each specific date
# If a date appears in >1 borough's top list, we flag it
date_counts <- ranked_days |>
  count(date, name = "borough_count")

# 3. Join the counts back to the data
plot_data <- ranked_days |>
  left_join(date_counts, by = "date") |>
  mutate(
    # Create the label for the legend
    event_type = if_else(borough_count > 1, "Shared Storm Event", "Local/Unique Event"),
    # Ensure date is treated as a factor for the x-axis
    date_label = factor(date)
  )

# 4. Plot
ggplot(plot_data, aes(x = date_label, y = mean_depth, fill = event_type)) +
  geom_col() +
  # Free x-axis allows each borough to show its own dates
  facet_wrap(~borough, scales = "free_x") + 
  # Custom colors: Dark for shared storms, Light for local ones
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
save_plot("shared_vs_local_storm_events.png", width = 12, height = 8)



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
  # Base Map: Clean grey style similar to the notebook
  addProviderTiles(providers$CartoDB.Positron) |> 
  
  # Add Circle Markers
  addCircleMarkers(
    ~longitude, ~latitude,
    # Scale radius by square root of depth for better visibility
    radius = ~sqrt(daily_mean) * 8, 
    
    # Styling
    color = ~pal(daily_mean),
    fillColor = ~pal(daily_mean),
    fillOpacity = 0.7,
    stroke = TRUE,
    weight = 1,
    
    # Popup: Click to see the specific Date and Depth
    popup = ~paste0(
      "<b>Sensor:</b> ", name_x, "<br>",
      "<b>Event Date:</b> ", deploy_date, "<br>",
      "<b>Daily Avg Depth:</b> ", round(daily_mean, 2), " inches<br>",
      "<b>Type:</b> ", deploy_type
    ),
    
    # Label: Hover to see Name and Depth
    label = ~paste0(name_x, ": ", round(daily_mean, 2), "\"")
  ) |>
  
  # Legend
  addLegend(
    "bottomright",
    pal = pal,
    values = ~daily_mean,
    title = "Max Daily Avg (in)",
    opacity = 1
  )

htmlwidgets::saveWidget(
  flood_map,
  file = file.path(PLOT_DIR, "peak_flood_events_map.html"),
  selfcontained = TRUE
)
message("Saved map: ", file.path(PLOT_DIR, "peak_flood_events_map.html"))

master_data <- master_data %>%
  rename(name = name_x)

street_ranking <- master_data |>
  filter(depth_inches > 0) |>
  group_by(name) |>
  summarise(avg_flood_depth = mean(depth_inches, na.rm = TRUE)) |>
  arrange(desc(avg_flood_depth)) |>
  slice_head(n = 20) # Keep the Top 20

ggplot(street_ranking, aes(x = avg_flood_depth, y = reorder(name, avg_flood_depth))) +
  geom_col(fill = "steelblue") +
  theme_classic() +
  labs(
    title = "Top 20 Streets with the Deepest Floods",
    subtitle = "Average recorded depth (inches) when flooding occurs",
    x = "Average Depth (inches)",
    y = NULL # Hides the 'name' label since the text is self-explanatory
  ) +
  theme(
    axis.text.y = element_text(size = 10), # Adjust text size for readability
    plot.title = element_text(face = "bold")
  )
save_plot("top20_most_flooded_intersections.png", width = 12, height = 8)

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
    total_flood_events = n(), # Useful context: how often does it happen?
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
save_plot("top20_rainiest_streets_aggregated.png", width = 12, height = 8)


street_data_all <- master_data |>
  pivot_longer(
    cols = c(street1, street2), 
    names_to = "street_pos", 
    values_to = "street_name"
  ) |>
  # Clean up empty or NA street names
  filter(!is.na(street_name) & street_name != "") |>
  mutate(street_name = str_trim(street_name))

# 2. ANALYZE (Include Zeros)
# We calculate the mean of 'depth_inches' across ALL records (dry + wet)
street_ranking_all <- street_data_all |>
  group_by(street_name) |>
  summarise(
    avg_depth_overall = mean(depth_inches, na.rm = TRUE),
    total_records = n(),
    .groups = "drop"
  ) |>
  arrange(desc(avg_depth_overall)) |>
  slice_head(n = 20)

# 3. PLOT
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
save_plot("top20_streets_chronic_flooding_all_days.png", width = 12, height = 8)
