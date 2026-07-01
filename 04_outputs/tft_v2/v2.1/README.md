# TFT v2.1 — Proper categorical encoding

Configuration:

- LLM filter: usable=1
- Categoricals: proper categorical encoding (event_type_primary, time_horizon with embedding tables of dim 5 and 3)
- Targets: single (log_volume)
- Prediction horizon: single (1h)
- Entity flags: not included
- Architecture: hidden_size=32, attention_head_size=4, dropout=0.1, hidden_continuous_size=16
- Early stopping patience: 5

Best val_loss: 0.274 at epoch 32, stopped at epoch 38.
Trainable parameters: 114,027.

Role in the project: Second variant in the ablation, isolating the effect of categorical encoding over v2.0. Demand_impact emerged as a top feature (43% importance) only with proper categorical encoding. Used in the ablation reported in Appendix [X]. Not the reported v2 model.
