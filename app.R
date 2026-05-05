library(jsonlite)
library(shiny)
library(xgboost)

serving_bundle <- fromJSON("artifacts/serving_bundle.json")
xgb_model <- xgb.load("artifacts/xgb_model.json")
metrics_table <- read.csv("outputs/metrics_comparison_table.csv", check.names = FALSE)
top_features_table <- read.csv("outputs/xgb_feature_importance.csv", check.names = FALSE)
input_schema <- split(serving_bundle$input_schema, seq_len(nrow(serving_bundle$input_schema)))
app_summary <- list(
  label_rule = serving_bundle$label_rule,
  model_name = serving_bundle$model_name,
  metrics = serving_bundle$metrics
)

field_labels <- c(
  fixed_acidity = "Fixed Acidity",
  volatile_acidity = "Volatile Acidity",
  citric_acid = "Citric Acid",
  residual_sugar = "Residual Sugar",
  chlorides = "Chlorides",
  free_sulfur_dioxide = "Free Sulfur Dioxide",
  total_sulfur_dioxide = "Total Sulfur Dioxide",
  density = "Density",
  ph = "pH",
  sulphates = "Sulphates",
  alcohol = "Alcohol"
)

field_help <- c(
  fixed_acidity = "Non-volatile acids that contribute to tartness.",
  volatile_acidity = "Acetic acid level; too high often means vinegar-like flavor.",
  citric_acid = "Adds freshness and structure.",
  residual_sugar = "Sugar remaining after fermentation.",
  chlorides = "Salt concentration in the wine.",
  free_sulfur_dioxide = "Active SO2 that protects wine from oxidation.",
  total_sulfur_dioxide = "Total SO2 including bound and free forms.",
  density = "Density of the wine sample.",
  ph = "Acidity/alkalinity scale.",
  sulphates = "Can contribute to preservation and mouthfeel.",
  alcohol = "Alcohol by volume percentage."
)

build_numeric_input <- function(item) {
  numericInput(
    inputId = item$name,
    label = field_labels[[item$name]],
    value = round(item$default, 3),
    min = floor(item$min * 100) / 100,
    max = ceiling(item$max * 100) / 100,
    step = item$step
  )
}

about_data_frame <- do.call(
  rbind,
  lapply(input_schema, function(item) {
    data.frame(
      Feature = field_labels[[item$name]],
      RawField = item$name,
      Default = round(item$default, 3),
      Min = round(item$min, 3),
      Max = round(item$max, 3),
      Description = field_help[[item$name]],
      check.names = FALSE
    )
  })
)

safe_scale <- function(values, centers, scales) {
  scaled <- (values - centers) / scales
  scaled[!is.finite(scaled)] <- 0
  scaled
}

build_engineered_features <- function(raw_values) {
  feature_scaled <- safe_scale(
    raw_values,
    serving_bundle$feature_scaler$mean,
    serving_bundle$feature_scaler$scale
  )

  pca_scores <- as.numeric(
    (matrix(feature_scaled - serving_bundle$pca_2d$mean, nrow = 1)) %*%
      t(serving_bundle$pca_2d$components)
  )

  cluster_centers <- serving_bundle$kmeans$cluster_centers
  distances <- apply(cluster_centers, 1, function(center) {
    sum((feature_scaled - center) ^ 2)
  })
  cluster_value <- which.min(distances) - 1

  free_to_total_so2 <- if (raw_values["total_sulfur_dioxide"] == 0) {
    0
  } else {
    as.numeric(raw_values["free_sulfur_dioxide"] / raw_values["total_sulfur_dioxide"])
  }

  alcohol_acidity_ratio <- if (raw_values["volatile_acidity"] == 0) {
    0
  } else {
    as.numeric(raw_values["alcohol"] / raw_values["volatile_acidity"])
  }

  c(
    cluster = cluster_value,
    free_to_total_SO2 = free_to_total_so2,
    total_acidity = as.numeric(raw_values["fixed_acidity"] + raw_values["volatile_acidity"]),
    alcohol_acidity_ratio = alcohol_acidity_ratio,
    PC1_score = pca_scores[1],
    PC2_score = pca_scores[2]
  )
}

predict_wine_quality_r <- function(sample_values) {
  raw_order <- serving_bundle$raw_feature_columns
  raw_values <- unlist(sample_values[raw_order], use.names = TRUE)
  raw_values <- as.numeric(raw_values)
  names(raw_values) <- raw_order

  engineered <- build_engineered_features(raw_values)
  model_values <- c(raw_values, engineered)[serving_bundle$model_feature_columns]
  scaled_model_values <- safe_scale(
    model_values,
    serving_bundle$model_scaler$mean,
    serving_bundle$model_scaler$scale
  )

  probability <- as.numeric(
    predict(xgb_model, matrix(scaled_model_values, nrow = 1))
  )
  predicted_class <- as.integer(probability >= serving_bundle$threshold)
  predicted_label <- if (predicted_class == 1) "Good" else "Bad"

  list(
    predicted_class = predicted_class,
    predicted_label = predicted_label,
    good_quality_probability = round(probability, 4),
    bad_quality_probability = round(1 - probability, 4),
    threshold = serving_bundle$threshold,
    message = if (predicted_class == 1) {
      "This wine is predicted to be good quality."
    } else {
      "This wine is predicted to be bad quality."
    },
    engineered_features = as.list(round(engineered, 4))
  )
}

ui <- navbarPage(
  "Red Wine Quality Predictor",
  header = tags$head(
    tags$style(HTML("
      .hero-box {padding: 18px; border-radius: 10px; background: #f6f8fb; margin-bottom: 18px;}
      .result-box {padding: 18px; border-radius: 10px; margin-top: 16px; background: #f8f9fa; border-left: 6px solid #2c7be5;}
      .metric-box {padding: 14px; border-radius: 8px; background: #f6f8fb; margin-bottom: 12px;}
      .probability-text {font-size: 18px; font-weight: 600;}
      .small-note {color: #5c6773;}
    "))
  ),
  tabPanel(
    "Prediction",
    fluidPage(
      div(
        class = "hero-box",
        h3("Predict whether a red wine is good or bad"),
        p("This app follows the current project pipeline and serves the exported tuned XGBoost model directly inside Shiny."),
        p(class = "small-note", "Only the 11 raw chemistry inputs are entered by the user. Cluster and PCA-based features are generated automatically in the backend.")
      ),
      sidebarLayout(
        sidebarPanel(
          width = 4,
          lapply(input_schema, build_numeric_input),
          actionButton("predict_btn", "Predict Wine Quality", class = "btn-primary")
        ),
        mainPanel(
          width = 8,
          uiOutput("prediction_result"),
          br(),
          h4("Auto-generated engineered features"),
          tableOutput("engineered_table")
        )
      )
    )
  ),
  tabPanel(
    "Model Overview",
    fluidPage(
      div(
        class = "hero-box",
        h3("Project summary"),
        p(sprintf("Label rule: %s", app_summary$label_rule)),
        p(sprintf("Serving model: %s", app_summary$model_name))
      ),
      fluidRow(
        column(
          4,
          div(class = "metric-box", h4("Accuracy"), p(app_summary$metrics$accuracy)),
          div(class = "metric-box", h4("Precision"), p(app_summary$metrics$precision))
        ),
        column(
          4,
          div(class = "metric-box", h4("Recall"), p(app_summary$metrics$recall)),
          div(class = "metric-box", h4("F1"), p(app_summary$metrics$f1))
        ),
        column(
          4,
          div(class = "metric-box", h4("ROC-AUC"), p(app_summary$metrics$roc_auc))
        )
      ),
      h4("Model comparison from the project"),
      tableOutput("metrics_table"),
      br(),
      h4("Top XGBoost features"),
      tableOutput("top_features_table")
    )
  ),
  tabPanel(
    "About Data",
    fluidPage(
      div(
        class = "hero-box",
        h3("Input fields"),
        p("The table below shows the raw variables accepted by the app, along with default values and the observed range from the cleaned project dataset.")
      ),
      tableOutput("about_data_table")
    )
  )
)

server <- function(input, output, session) {
  prediction <- reactiveVal(NULL)

  observeEvent(input$predict_btn, {
    sample_values <- lapply(input_schema, function(item) {
      input[[item$name]]
    })
    names(sample_values) <- vapply(input_schema, function(item) item$name, character(1))

    prediction_result <- predict_wine_quality_r(sample_values)
    prediction(prediction_result)
  })

  output$prediction_result <- renderUI({
    result <- prediction()
    if (is.null(result)) {
      return(
        div(
          class = "hero-box",
          h4("Waiting for input"),
          p("Fill in the wine chemistry values on the left and click the predict button.")
        )
      )
    }

    color <- if (isTRUE(result$predicted_class == 1)) "#1f9d55" else "#d64545"
    div(
      class = "result-box",
      style = paste0("border-left-color:", color, ";"),
      h3(sprintf("Prediction: %s wine", result$predicted_label)),
      p(class = "probability-text", sprintf("Good wine probability: %.1f%%", result$good_quality_probability * 100)),
      p(sprintf("Bad wine probability: %.1f%%", result$bad_quality_probability * 100)),
      p(result$message),
      p(class = "small-note", sprintf("Classification threshold: %.2f", result$threshold))
    )
  })

  output$engineered_table <- renderTable({
    result <- prediction()
    if (is.null(result)) {
      return(NULL)
    }

    engineered <- result$engineered_features
    data.frame(
      Feature = names(engineered),
      Value = round(unlist(engineered), 4),
      check.names = FALSE
    )
  }, striped = TRUE, bordered = TRUE, spacing = "m")

  output$metrics_table <- renderTable({
    metrics_table
  }, striped = TRUE, bordered = TRUE, spacing = "m", digits = 3)

  output$top_features_table <- renderTable({
    if (nrow(top_features_table) == 0) {
      return(data.frame(Message = "Run model_interpretation.py to populate feature importance output."))
    }
    head(top_features_table, 5)
  }, striped = TRUE, bordered = TRUE, spacing = "m", digits = 2)

  output$about_data_table <- renderTable({
    about_data_frame
  }, striped = TRUE, bordered = TRUE, spacing = "m", digits = 3)
}

shinyApp(ui = ui, server = server)
