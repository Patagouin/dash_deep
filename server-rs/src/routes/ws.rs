use axum::extract::ws::{WebSocketUpgrade, Message};
use axum::{response::IntoResponse, extract::State};
use futures_util::StreamExt;
use crate::AppState;

pub async fn ws_handler(State(_state): State<std::sync::Arc<AppState>>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(|mut socket| async move {
        let _ = socket.send(Message::Text("connected".into())).await;
        while let Some(Ok(msg)) = socket.next().await {
            match msg {
                Message::Text(t) => { let _ = socket.send(Message::Text(format!("echo: {}", t))).await; }
                Message::Binary(_b) => { let _ = socket.send(Message::Text("binary not supported".into())).await; }
                Message::Ping(p) => { let _ = socket.send(Message::Pong(p)).await; }
                Message::Pong(_) => {}
                Message::Close(_) => break,
            }
        }
    })
}
