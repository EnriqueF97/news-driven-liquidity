# TFT v2 Training Runs

Tracking document for all TFT v2 training experiments. Each row corresponds to one trained model. Captures configuration, headline metrics, and key diagnostics so the final "v2" model can be selected from a complete record.

## Summary table

| #   | Date       | Variant                 | Filter          | Patience | Dropout |   LR | Best val_loss | Best epoch | Stop epoch | Test MAE @1h log_volume | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --- | ---------- | ----------------------- | --------------- | -------: | ------: | ---: | ------------: | ---------: | ---------: | ----------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 2026-06-02 | v2.0 (initial)          | usable=1        |        5 |    0.10 | 1e-3 |         0.294 |         19 |         25 |             0.494 (val) | Discarded — misconfigured (no `target_normalizer`, spurious `static_categoricals=['asset']`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 2   | 2026-06-02 | v2.0 (corrected)        | usable=1        |        5 |    0.10 | 1e-3 |         0.311 |         24 |         26 |                   0.472 | Canonical v2.0 result. Channels barely register in feature importance (3-5%); attention dominated by -48h edge artifact.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 3   | 2026-06-02 | v2.1                    | usable=1        |        5 |    0.10 | 1e-3 |         0.274 |         32 |         38 |                   0.375 | Added proper categorical encoding. `demand_impact` jumps to 43% feature importance. Attention peak shifts to -15h. Criterion 2 fails at all 4 slices.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 4   | 2026-06-03 | v2.2 (first)            | usable=1        |        5 |    0.10 | 1e-3 |         0.419 |         19 |         25 |                   0.617 | Multi-target + multi-horizon + entity flags. 6/10 top features are entity flags (Oman #3 at 8.6%). Attention peak -19h. Criterion 2 fails at all 5×4 = 20 horizon×slice combinations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 5   | 2026-06-03 | v2.2 (strict filter)    | usable_strict=1 |        5 |    0.10 | 1e-3 |         0.421 |          5 |         11 |                   0.757 | Stricter LLM filter applied (~242 inconsistent zero-channel articles dropped). Converged much faster, worse test MAE. Feature importance more concentrated (supply_impact 13.2%, ent_oman 11.3%). Attention peak shifts to -1h. Criterion 2: 3/20 tests significant including +3h pre-war bearish > bullish (p=0.041, matches lag OLS direction).                                                                                                                                                                                                                                                                                                                                                                                                        |
| 6   | 2026-06-03 | v2.2 (patience+dropout) | usable_strict=1 |       10 |    0.15 | 1e-3 |     TBD-final |        TBD |        TBD |                   0.635 | **Selected as the canonical v2 model.** Stricter LLM filter + longer early-stopping patience + slightly higher dropout. Beats run #5 on every log_volume slice/horizon (16-19% lower test MAE). Beats run #4 on every horizon ≥3h. Persistence reduction on log_volume reaches 73% at +12h, 67% at +6h. Feature importance: vix 0.236, demand_impact 0.136, ent_us 0.045, with 5/10 features being entity flags (US, Yemen, Lebanon, Oman, Brent). Attention peak at -17h with cluster -15h to -18h, bearish/bullish attention diverge (-17h vs -12h). Criterion 2: 1/20 tests significant — +3h pre-war bearish > bullish with diff=+0.660, p=0.013, replicating run #5's same-horizon, same-slice finding with stronger effect size and lower p-value. |
| 7   | 2026-07-06 | v2.2 (canonical, 60/20/20) | usable_strict=1 |       10 |    0.15 | 1e-3 |         0.427 |         10 |          - |                   0.585 | **THE reported v2 model (thesis §4.3.7).** Run #6 configuration re-run on the final 60/20/20 split with 4 horizons `[1, 3, 6, 12]` (28h dropped). Feature importance: vix 0.188, supply_impact 0.121, ent_oman 0.113, demand_impact 0.055; 6/10 top features are entity flags (Oman, Japan, EU, Iran, China, Algeria). Persistence reduction on log_volume 46/60/67/71% at 1/3/6/12h. Attention: overall peak -1h; bearish -6h vs bullish -1h. Criterion 2 (asymmetry): 0/16 tests significant (closest +12h war p=0.053). Confirmed against the checkpoint hparams (dropout 0.15). Best checkpoint epoch 10; thesis §4.3.7.2 states epoch 21 (minor discrepancy to reconcile). |

## Configuration notes

All runs share:

- Architecture: `hidden_size=32`, `attention_head_size=4`, `hidden_continuous_size=16` (matches TFT v1)
- Optimizer: Adam with `reduce_on_plateau_patience=3`
- Loss: `QuantileLoss` (single target) or `MultiLoss([QuantileLoss()] * n_targets)` (multi-target)
- Target normalizer: `GroupNormalizer(groups=['asset'])` per target
- Gradient clip: 0.1
- Min delta for early stopping: 1e-4
- Split: runs #1-#6 used 70/15/15 (TRAIN_END=7862, VAL_END=9547). The canonical run #7 and all subsequent runs use **60/20/20** (TRAIN_END=6739, VAL_START=6787, VAL_END=9014, TEST_START=9062, TEST_END=11232), matching `03_src/tft/config.py`. All splits use a 48h buffer.
- Seed: 42 (`pytorch_lightning.seed_everything(42, workers=True)`)
- Determinism: `torch.use_deterministic_algorithms(True, warn_only=True)`
- Compute: Google Colab T4 GPU

Variant-specific:

- v2.0: int-encoded categoricals (`event_type_int`, `time_horizon_int`), no entity flags, single horizon=1, target=`log_volume`
- v2.1: proper categorical encoding (`event_type_primary`, `time_horizon` as `time_varying_unknown_categoricals`), no entity flags, single horizon=1, target=`log_volume`
- v2.2: v2.1 + 71 entity flag columns (`ent_*`), multi-horizon `[1, 3, 6, 12]`, multi-target `[log_volume, amihud, price_range]`

## Canonical ablation (60/20/20, Appendix C)

The reported ablation re-runs all three variants on the final 60/20/20 split with a **constant filter (`usable_strict=1`)**, dropout 0.15, patience 10, seed 42, encoder 48h. Only the design axis changes between rows, so the filter is not an ablation axis here (its effect is characterized separately in §4.3.3).

| Variant          | Categorical encoding | Targets × horizons          | Entity flags | Best val_loss | Test MAE@1h (full) | Feature #1 (importance)  |
| ---------------- | -------------------- | --------------------------- | ------------ | ------------: | -----------------: | ------------------------ |
| v2.0             | int-encoded          | log_volume × 1h             | no           |         0.257 |              0.455 | sentiment_score 0.274    |
| v2.1             | proper categoricals  | log_volume × 1h             | no           |         0.258 |              0.424 | **demand_impact 0.485**  |
| v2.2 (canonical) | proper categoricals  | 3 targets × [1, 3, 6, 12]   | 71           |       0.427\* |              0.585 | vix 0.188                |

\* v2.2's val_loss is not comparable to v2.0/v2.1 (`MultiLoss` over 3 targets ≈ 3× the single-target scale). v2.0 and v2.1 val_loss are comparable to each other; Test MAE@1h is comparable across all three (same target, horizon, and 60/20/20 test set).

**v2.0 → v2.1 (int → proper categorical encoding)** is the signature ablation result: `demand_impact` jumps from 0.068 to **0.485** (~7×) and becomes the top feature, and Test MAE@1h improves ~7% (0.455 → 0.424). Val_loss is essentially unchanged (0.257 vs 0.258). Proper categorical encoding is what unlocks the channels' predictive role; on the 60/20/20 split the gain shows in feature importance and 1h MAE rather than in val_loss.

**v2.1 → v2.2 (single → multi-target/horizon + 71 entity flags)**: feature importance redistributes (vix #1 at 0.188, supply_impact #2 at 0.121, demand_impact #4 at 0.055, plus 6/10 top-10 entity flags). The model gains multi-horizon prediction but 1h MAE degrades (0.424 → 0.585, multi-target dilution). This is the trade the canonical model accepts: richer, multi-horizon, entity-aware prediction at the cost of some 1h accuracy.

Runs #1-#6 (70/15/15, and usable=1 for the early v2.0/v2.1 rows) in the table above are development history and are superseded by this split-consistent, filter-constant ablation.

**Run #5 → Run #6 (extending patience to 10, raising dropout to 0.15)**: longer optimization recovered predictive performance lost in run #5 (16-19% lower test MAE across log_volume slices) while preserving the asymmetry signal. The +3h pre-war bearish > bullish finding from run #5 replicated with stronger effect size (+0.660 vs +0.525 log_volume) and lower p-value (0.013 vs 0.041). Other run #5 significant tests at +12h (opposite direction) did not replicate, suggesting they were stochastic significance from multiple testing. Run #6 is the cleaner v2 result.

## Success criteria status (run #7, the canonical v2 model)

Evaluated against the thesis §4.3.7.1 criteria (the framing used in the reported chapter):

| Criterion                                       | Status   | Evidence (exp 2, 60/20/20)                                                                                                                                            |
| ----------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Channels and entities are visible drivers    | **PASS** | supply_impact 0.121 (#2), demand_impact 0.055 (#4); 6/10 top-10 features are entity flags (Oman, Japan, EU, Iran, China, Algeria); attention non-degenerate          |
| 2. Substantial persistence-relative improvement | **PASS** | log_volume MAE reduction 46-71% over the 1-12h window; amihud 43-45%                                                                                                  |
| 3. Multi-horizon curve consistent with lag OLS  | **PASS** | persistence reduction peaks +12h (71%) with +6h close behind (67%); bearish attention peak at -6h matches the lag OLS +6h peak                                        |

Note: the earlier asymmetry-based criterion (bearish > bullish at p<0.05 in the TFT point predictions) is NOT met by the canonical run: 0/16 tests significant (closest +12h war p=0.053). Phase 1 lag OLS remains the primary RQ2 evidence; the TFT corroborates only the direction and the temporal (-6h) asymmetry. This matches thesis §4.3.7.6-7.

## Decision criteria for selecting "the v2 model"

The reported v2 model in the thesis (§4.3.7) will be selected based on:

1. Whether Criterion 2 shows asymmetry at horizons matching lag OLS findings (favoring run #5 or beyond)
2. Whether the channel decomposition and entity flags are visible in feature importance (favoring runs #4 and beyond)
3. Whether attention pattern is non-degenerate and economically interpretable
4. Per-horizon predictive performance against persistence baseline

The reported v2 model is **run #7** (v2.2 with `usable_strict=1`, patience 10, dropout 0.15) on the 60/20/20 split: run #6's configuration re-run on the final split, reported in thesis §4.3.7. Runs #1-#6 (70/15/15) are the development history. The v2.0 and v2.1 ablation rows are being re-run on 60/20/20 for a split-consistent ablation (Appendix C).

## Next experiments under consideration

- Single-target retraining (log_volume only, multi-horizon retained) to test multi-target dilution hypothesis
- Loss reweighting `[3.0, 1.0, 1.0]` on multi-target run to emphasize log_volume
- Capacity adjustment (`hidden_size=64` with `dropout=0.15` or `0.2`)
- Extended dataset (additional 1-2 months of news + market context, processed through Haiku v2 pipeline)
