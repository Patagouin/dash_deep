use axum::{extract::State, http::StatusCode, Json};
use crate::{AppState, CorrRequest, CorrResponse};

pub async fn correlation_handler(State(_state): State<std::sync::Arc<AppState>>, Json(_req): Json<CorrRequest>) -> Result<Json<CorrResponse>, (StatusCode, String)> {
    // Stub: renvoie 0.0
    Ok(Json(CorrResponse { correlation: 0.0 }))
}
