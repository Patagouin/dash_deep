use axum::{extract::State, http::StatusCode, Json};
use serde_json::json;
use crate::{AppState, StatsRequest, StatsResponse};

pub async fn stats_handler(State(_state): State<std::sync::Arc<AppState>>, Json(_req): Json<StatsRequest>) -> Result<Json<StatsResponse>, (StatusCode, String)> {
    Ok(Json(StatsResponse { result: json!({ "ok": true }) }))
}
