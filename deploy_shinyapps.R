required_vars <- c("SHINYAPPS_NAME", "SHINYAPPS_TOKEN", "SHINYAPPS_SECRET")
missing_vars <- required_vars[Sys.getenv(required_vars, unset = "") == ""]

if (length(missing_vars) > 0) {
  stop(
    sprintf(
      "Missing shinyapps.io credentials: %s",
      paste(missing_vars, collapse = ", ")
    )
  )
}

if (!requireNamespace("rsconnect", quietly = TRUE)) {
  install.packages("rsconnect", repos = "https://cloud.r-project.org")
}

library(rsconnect)

app_name <- Sys.getenv("SHINYAPP_APPNAME", unset = "red-wine-quality-predictor")
app_title <- Sys.getenv("SHINYAPP_TITLE", unset = "Red Wine Quality Predictor")
app_files <- c(
  "app.R",
  "artifacts/serving_bundle.json",
  "artifacts/xgb_model.json",
  "outputs/metrics_comparison_table.csv",
  "outputs/xgb_feature_importance.csv"
)

rsconnect::setAccountInfo(
  name = Sys.getenv("SHINYAPPS_NAME"),
  token = Sys.getenv("SHINYAPPS_TOKEN"),
  secret = Sys.getenv("SHINYAPPS_SECRET")
)

rsconnect::deployApp(
  appDir = ".",
  appFiles = app_files,
  appName = app_name,
  appTitle = app_title,
  appPrimaryDoc = "app.R",
  forceUpdate = TRUE
)
