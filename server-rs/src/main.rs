use axum::{routing::{get, post}, Router};
use axum::extract::State;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::{cors::{Any, CorsLayer}, trace::TraceLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod routes;
use routes::{health::health_handler, ws::ws_handler, predict::predict_handler, simulate::simulate_handler, stats::stats_handler, corr::correlation_handler};

#[derive(Clone, Default)]
struct AppState {}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into())))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cors = CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any);

    let app_state = Arc::new(AppState::default());
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/ws", get(ws_handler))
        .route("/api/v1/predict", post(predict_handler))
        .route("/api/v1/simulate", post(simulate_handler))
        .route("/api/v1/stats", post(stats_handler))
        .route("/api/v1/correlation", post(correlation_handler))
        .with_state(app_state)
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    let port = std::env::var("PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8080);
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::UNSPECIFIED, port)).await.unwrap();
    tracing::info!(port=%port, "server listening");
    axum::serve(listener, app).await.unwrap();
}

#[derive(Debug, Deserialize)]
struct PredictRequest {
    inputs: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct PredictResponse {
    y_pred: serde_json::Value,
    latency_ms: u128,
}

#[derive(Debug, Deserialize)]
struct SimulateRequest {
    params: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct SimulateResponse {
    equity: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct StatsRequest {
    query: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct StatsResponse {
    result: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct CorrRequest {
    a: String,
    b: String,
}

#[derive(Debug, Serialize)]
struct CorrResponse {
    correlation: f64,
}
