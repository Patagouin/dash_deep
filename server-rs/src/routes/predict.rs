use std::time::Instant;
use axum::{extract::State, http::StatusCode, Json};
use serde_json::json;
use crate::{AppState, PredictRequest, PredictResponse};

pub async fn predict_handler(State(_state): State<std::sync::Arc<AppState>>, Json(_req): Json<PredictRequest>) -> Result<Json<PredictResponse>, (StatusCode, String)> {
    let t0 = Instant::now();
    // TODO: appeler un moteur de prédiction natif (plus tard ONNX/TensorRT)
    let y_pred = json!({ "values": [0.0] });
    let latency_ms = t0.elapsed().as_millis();
    Ok(Json(PredictResponse { y_pred, latency_ms }))
}
