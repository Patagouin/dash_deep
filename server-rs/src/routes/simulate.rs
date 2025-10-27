use axum::{extract::State, http::StatusCode, Json};
use serde_json::json;
use crate::{AppState, SimulateRequest, SimulateResponse};

pub async fn simulate_handler(State(_state): State<std::sync::Arc<AppState>>, Json(_req): Json<SimulateRequest>) -> Result<Json<SimulateResponse>, (StatusCode, String)> {
    // TODO: simulation rapide native
    Ok(Json(SimulateResponse { equity: json!({ "equity": [] }) }))
}
