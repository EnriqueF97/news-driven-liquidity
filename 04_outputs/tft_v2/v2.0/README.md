# TFT v2.0 — Initial Phase 2 baseline

Configuration:

- LLM filter: usable=1
- Categoricals: int-encoded (event_type_int, time_horizon_int)
- Targets: single (log_volume)
- Prediction horizon: single (1h)
- Entity flags: not included
- Architecture: hidden_size=32, attention_head_size=4, dropout=0.1, hidden_continuous_size=16
- Early stopping patience: 5

Best val_loss: 0.311 at epoch 24, stopped at epoch 26.
Trainable parameters: 117,935.

Role in the project: Phase 2 baseline using v1-style architectural choices with the Phase 2 feature set. Used in the ablation reported in Appendix [X]. Not the reported v2 model.
