# Decision log

One line per decision that would be hard to reconstruct later. The dissertation's
methodology chapter is written from this file.

| Date | Decision | Why |
|---|---|---|
| 2026-10-14 | Scope confirmed at three research questions | Supervisor feedback: scope is fine, keep all three |
| 2026-10-16 | FP32 is the control; macro-F1 held within one point | Isolates calibration change from accuracy change (internal validity) |
| 2026-10-16 | Metrics implemented in-repo rather than imported | Binning scheme and "delivered precision" must be explicit and testable |
