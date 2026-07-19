library(scoringutils)
library(dplyr)

d = read.csv("./multi_model_ensemble_weights/component_model_forecasts.csv")

d <- d %>% rename("quantile_level" = "output_type_id")

forecast_quantile <- d |>
  as_forecast_quantile(
    forecast_unit = c(
      "location", "reference_date", "target_end_date", "model", "horizon"
    )
  )
scores <- forecast_quantile |> score()

write.csv(scores, "./multi_model_ensemble_weights/scores.csv")
