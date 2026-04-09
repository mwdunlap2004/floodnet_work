library('tidyverse')
library('jsonlite')
library('leaflet')
library('htmltools')
rain_data = read.csv('got_rain.csv')
full_data=read.csv('full_data.csv')
full_data <- full_data |>
  mutate(borough = str_trim(borough)) 
rain_data <- rain_data |> 
  mutate(borough = str_trim(borough)) 

master_data <- read.csv('master_data.csv')


rain_data |>
  group_by(borough) |>
  summarise(mean_precip = mean(depth_inches, na.rm = TRUE)) |>
  ggplot(aes(x = borough, y = mean_precip)) +
  geom_col() +theme_classic()



full_data |>
  group_by(borough) |>
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE)) |>
  ggplot(aes(x = borough, y = mean_depth)) +
  geom_col() +theme_classic()


full_data |> group_by(date) |> 
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE)) |>
  slice_max(mean_depth, n = 10) |> 
  ggplot(aes(x = date, y = mean_depth)) +
  geom_col() +theme_classic()

full_data |>
  ggplot(aes(x = date, y = depth_inches, color = borough)) +
  geom_line() +
  theme_classic()

full_data |>
  group_by(date, borough) |> 
  summarise(mean_depth = mean(depth_inches, na.rm = TRUE)) |>
  ggplot(aes(x = date, y = mean_depth, color = borough)) +
  geom_line() +
  theme_classic()

df_plot <- full_data |>
  group_by(date, borough) |>
  summarise(mean_depth = mean(depth_inches), .groups = "drop")
df_plot$date <- as.Date(df_plot$date)
ggplot(df_plot, aes(x = date, y = mean_depth, color = borough, group = borough)) +
  geom_line()+theme_classic()



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

plot_data <- rainiest_days |>
  group_by(date, borough) |>
  summarise(mean_precip = mean(depth_inches, na.rm = TRUE), .groups = "drop")

ggplot(plot_data, aes(x = factor(date), y = mean_precip, fill = borough)) +
  geom_col(position = "dodge") +
  theme_classic() +
  labs(title = "Rainiest Selected Days by Borough", x = "Date", y = "Mean Precipitation") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


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

leaflet(data = top_gage_events) |>
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