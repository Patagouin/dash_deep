# -*- coding: utf-8 -*-
"""
Callbacks Keras réutilisables pour l'entraînement de modèles ML.
Utilisables dans playground.py, prediction_callbacks/training.py, etc.
"""

import time
import logging
import plotly.graph_objs as go
from dash import html, dash_table
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import Callback as KerasCallback
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    KerasCallback = object
    logging.warning("TensorFlow non disponible pour les callbacks")


class TrainingProgressCallback:
    """
    Callback de progression d'entraînement générique.
    Peut être utilisé avec set_progress de Dash ou directement.
    
    Usage:
        callback = TrainingProgressCallback(
            set_progress=set_progress,  # Fonction Dash set_progress
            total_epochs=100,
            update_interval=0.5
        )
        model.fit(..., callbacks=[callback])
    """
    
    def __init__(
        self,
        set_progress=None,
        total_epochs: int = 100,
        update_interval: float = 0.5,
        socketio=None,
        stage: str = 'training'
    ):
        """
        Args:
            set_progress: Fonction Dash pour mettre à jour la progression
            total_epochs: Nombre total d'epochs prévu
            update_interval: Intervalle minimum entre les mises à jour (secondes)
            socketio: Instance SocketIO pour émission (optionnel)
            stage: Nom de la phase ('training', 'tuner', 'final')
        """
        self.set_progress = set_progress
        self.total_epochs = total_epochs
        self.update_interval = update_interval
        self.socketio = socketio
        self.stage = stage
        
        # Métriques
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        
        # Timing
        self.last_update = time.time()
        self.current_epoch = 0
    
    def reset(self):
        """Réinitialise les métriques."""
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.current_epoch = 0
    
    def set_stage(self, stage: str, total_epochs: int = None):
        """Change la phase d'entraînement."""
        self.stage = stage
        if total_epochs is not None:
            self.total_epochs = total_epochs
        self.reset()
    
    def on_train_begin(self, logs=None):
        """Appelé au début de l'entraînement."""
        self.reset()
        self._emit_progress(0, "Démarrage de l'entraînement...")
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Appelé à la fin de chaque epoch.
        Collecte les métriques et met à jour l'UI si nécessaire.
        """
        logs = logs or {}
        
        # Collecter les métriques
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        # Accuracy/DA (plusieurs noms possibles)
        acc = (
            logs.get('accuracy') or 
            logs.get('directional_accuracy') or 
            logs.get('main_output_directional_accuracy')
        )
        val_acc = (
            logs.get('val_accuracy') or 
            logs.get('val_directional_accuracy') or 
            logs.get('val_main_output_directional_accuracy')
        )
        
        if loss is not None:
            self.losses.append(float(loss))
        if val_loss is not None:
            self.val_losses.append(float(val_loss))
        if acc is not None:
            self.accuracies.append(float(acc))
        if val_acc is not None:
            self.val_accuracies.append(float(val_acc))
        
        self.current_epoch = epoch + 1
        
        # Mettre à jour l'UI si assez de temps est passé ou si c'est le dernier epoch
        now = time.time()
        is_last = (self.current_epoch >= self.total_epochs)
        
        if (now - self.last_update > self.update_interval) or is_last:
            self.last_update = now
            progress = self.current_epoch / max(1, self.total_epochs)
            loss_str = f"{loss:.4f}" if loss is not None else "?"
            msg = f"Epoch {self.current_epoch}/{self.total_epochs} - Loss={loss_str}"
            self._emit_progress(progress, msg)
    
    def _emit_progress(self, progress: float, message: str):
        """Émet la progression vers l'UI."""
        if self.set_progress is not None:
            fig = self._build_history_figure()
            progress_html = self._build_progress_html(progress, message)
            try:
                self.set_progress((progress_html, fig))
            except Exception as e:
                logging.warning(f"Erreur émission progress: {e}")
        
        if self.socketio is not None:
            try:
                self.socketio.emit('update_terminal', {
                    'output': f"[{self.stage.upper()}] {message}\n"
                }, broadcast=True)
                self.socketio.emit('update_progress', {
                    'progress': int(progress * 100)
                }, broadcast=True)
            except Exception:
                pass
    
    def _build_progress_html(self, progress: float, message: str):
        """Construit le HTML de la barre de progression."""
        pct = int(progress * 100)
        return html.Div([
            html.Div(message, style={'marginBottom': '8px', 'color': '#4CAF50'}),
            html.Div([
                html.Div(style={
                    'width': f'{pct}%',
                    'height': '10px',
                    'backgroundColor': '#4CAF50',
                    'transition': 'width 0.2s'
                }),
            ], style={
                'width': '100%',
                'height': '10px',
                'backgroundColor': '#555',
                'borderRadius': '4px',
                'overflow': 'hidden'
            })
        ])
    
    def _build_history_figure(self) -> go.Figure:
        """Construit la figure d'historique d'entraînement."""
        fig = go.Figure()
        
        # Tracer Loss (axe gauche)
        if self.losses:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(self.losses) + 1)),
                y=self.losses,
                mode='lines+markers',
                name='Loss train',
                line={'color': '#2ca02c', 'width': 2},
                marker={'size': 6},
                yaxis='y'
            ))
        
        if self.val_losses:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(self.val_losses) + 1)),
                y=self.val_losses,
                mode='lines+markers',
                name='Loss val',
                line={'color': '#d62728', 'width': 2},
                marker={'size': 6},
                yaxis='y'
            ))
        
        # Tracer Accuracy/DA (axe droit)
        if self.accuracies:
            accs_pct = [a * 100 for a in self.accuracies]
            fig.add_trace(go.Scatter(
                x=list(range(1, len(accs_pct) + 1)),
                y=accs_pct,
                mode='lines+markers',
                name='DA train %',
                line={'color': '#1f77b4', 'width': 2, 'dash': 'dot'},
                marker={'size': 6},
                yaxis='y2'
            ))
        
        if self.val_accuracies:
            vaccs_pct = [a * 100 for a in self.val_accuracies]
            fig.add_trace(go.Scatter(
                x=list(range(1, len(vaccs_pct) + 1)),
                y=vaccs_pct,
                mode='lines+markers',
                name='DA val %',
                line={'color': '#ff7f0e', 'width': 2, 'dash': 'dot'},
                marker={'size': 6},
                yaxis='y2'
            ))
        
        # Layout avec deux axes Y
        loss_info = ''
        if self.losses or self.val_losses:
            all_loss = [l for l in (self.losses + self.val_losses) if l is not None and l > 0]
            if all_loss:
                current_loss = float(all_loss[-1])
                if current_loss < 0.001:
                    loss_info = f' (actuel: {current_loss:.2e})'
                else:
                    loss_info = f' (actuel: {current_loss:.6f})'
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#000000',
            plot_bgcolor='#000000',
            font={'color': '#FFFFFF'},
            title=f'📊 Loss{loss_info} & DA',
            height=300,
            uirevision='training_hist',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                font={'size': 10}
            ),
            margin=dict(t=60, b=40, l=60, r=60),
            yaxis={'title': 'Loss', 'side': 'left', 'type': 'log'},
            yaxis2={
                'title': 'DA %',
                'overlaying': 'y',
                'side': 'right',
                'range': [0, 100],
                'ticksuffix': '%'
            }
        )
        
        return fig
    
    def get_final_metrics(self) -> dict:
        """Retourne les métriques finales."""
        import numpy as np
        
        return {
            'final_loss': self.losses[-1] if self.losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'final_accuracy': self.accuracies[-1] if self.accuracies else None,
            'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else None,
            'mean_accuracy': float(np.mean(self.accuracies)) if self.accuracies else None,
            'mean_val_accuracy': float(np.mean(self.val_accuracies)) if self.val_accuracies else None,
            'epochs_completed': self.current_epoch,
        }


if TF_AVAILABLE:
    class KerasProgressCallback(tf.keras.callbacks.Callback, TrainingProgressCallback):
        """
        Version Keras du callback de progression.
        Hérite de tf.keras.callbacks.Callback pour être utilisable directement avec model.fit().
        """
        
        def __init__(self, set_progress=None, total_epochs=100, update_interval=0.5, socketio=None, stage='training', max_trials=None):
            tf.keras.callbacks.Callback.__init__(self)
            TrainingProgressCallback.__init__(
                self,
                set_progress=set_progress,
                total_epochs=total_epochs,
                update_interval=update_interval,
                socketio=socketio,
                stage=stage
            )
            self.max_trials = max_trials
            self.trials_results = []
        
        def on_trial_end(self, trial):
            """
            Callback pour Keras Tuner - appelé à la fin de chaque trial.
            Affiche un tableau des résultats des trials.
            """
            trial_data = {
                'trial_id': trial.trial_id,
                'hyperparameters': str(trial.hyperparameters.values),
                'score': trial.score,
                'status': trial.status
            }
            self.trials_results.append(trial_data)

            results_df = pd.DataFrame(self.trials_results)
            results_df.rename(columns={
                'trial_id': 'Essai', 'hyperparameters': 'Hyperparamètres',
                'score': 'Score', 'status': 'Statut'
            }, inplace=True)

            table = dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in results_df.columns],
                data=results_df.to_dict('records'),
                style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
            )

            # Barre de progression
            percent = 0
            if self.max_trials:
                num_done = len(self.trials_results)
                percent = int((num_done / self.max_trials) * 100) if self.max_trials else 0

            progress_children = html.Div([
                html.Div(f'Recherche des hyperparamètres: {len(self.trials_results)}/{self.max_trials or "?"}', 
                        style={'marginBottom': '8px', 'color': '#4CAF50'}),
                html.Div([
                    html.Div(style={'width': f'{percent}%', 'height': '10px', 'backgroundColor': '#4CAF50', 'transition': 'width 0.2s'}),
                ], style={'width': '100%', 'height': '10px', 'backgroundColor': '#555', 'borderRadius': '4px', 'overflow': 'hidden'}),
                html.Div(table, style={'marginTop': '10px'})
            ])

            # Émettre la progression
            if self.set_progress is not None:
                fig = self._build_history_figure()
                try:
                    self.set_progress((progress_children, fig))
                except Exception:
                    pass
            
            # Émettre via socketio si disponible
            if self.socketio is not None:
                try:
                    self.socketio.emit('update_terminal', {
                        'output': f"[TRIAL] Trial {trial.trial_id} terminé — Score: {trial.score}\n"
                    }, broadcast=True)
                    self.socketio.emit('update_progress', {'progress': percent}, broadcast=True)
                except Exception:
                    pass

else:
    # Fallback si TensorFlow n'est pas disponible
    KerasProgressCallback = TrainingProgressCallback


# Alias pour compatibilité avec prediction_callbacks/training.py
DashProgressCallback = KerasProgressCallback

