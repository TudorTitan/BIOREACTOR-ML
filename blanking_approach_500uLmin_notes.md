# Empty-casing blanking check at 500 µL/min

## Available for a first-pass test

- Three experimental empty-casing acetone blank runs at 500 µL/min.
- Empty-casing CFD RTD (`rtd_curve (1).csv`).
- Gyroid CFD RTD (`rtd_curve.csv`).
- Three 30% Gyroid and three 40% Gyroid experimental runs at 500 µL/min.
- The nominal tracer loop volume is 7.5 µL. At 500 µL/min, that corresponds to a nominal pulse duration of 0.9 s (`volume / flow`), before dispersion in the valve and tubing.

This is enough to try the blank-based correction once at 500 µL/min. The existing blank and reactor data should only be combined if the valve, tubing, detector, flow rate, tracer preparation, and data processing were the same.

## What the first comparison says

The experimental blank is much broader than empty-casing CFD. After aligning each curve at its detected rise and normalizing each curve to unit area, the blank mean is about 33.1 s with SD 18.3 s; empty-casing CFD is about 6.6 s with SD 6.9 s. Those are onset-relative statistics, not absolute injection-to-detector times.

The difference in means is about 26 s. A variance-only estimate gives about 17 s of additional spread from the experimental system. These are rough moment estimates, not a validated equipment response curve.

The pointwise blank curve must not be subtracted from a Gyroid curve. For sequential flow elements, RTDs combine by convolution. The intended comparison is:

1. Infer the external system response from `experimental blank ≈ empty-casing CFD * external response`.
2. Predict the experimental Gyroid trace as `Gyroid CFD * external response`.
3. Compare the predicted detector trace to the measured Gyroid trace, using consistent pulse input and time alignment.

I attempted a nonnegative, smoothed deconvolution as a diagnostic. Its reconstructed blank still has an RMSE around 16% of the blank peak and produces a visibly unstable Gyroid prediction. I am not treating that corrected curve as a result. The curves differ so strongly in shape that this first fit is not reliable enough to use as a correction.

## Prepared diagnostic files

- `empty_casing_blank_vs_CFD_500uLmin.png`: shape overlay; each line is normalized to its own peak.
- `empty_casing_blank_vs_CFD_500uLmin.csv`: aligned blank and empty-casing CFD values used for the plot.
- `gyroid_RTD_overlay_500uLmin_30pct_40pct.png` and matching CSV: the earlier raw Gyroid CFD versus experimental comparison, before applying an apparatus response.

## What is still needed

For another 500 µL/min analysis, no new data are essential if the blank runs were collected with exactly the same setup and the 7.5 µL loop was switched in the same way. Please confirm that the three blank runs used the same valve, tubing lengths/IDs, flow cell, and detector acquisition settings as the Gyroid runs. The absolute valve-switch time (or a signal/record of that event) would also let us compare absolute transit times instead of aligning by detected curve onset.

To compare at other flow rates, collect at least three empty-casing blank runs at each matching flow rate, with the same 7.5 µL tracer loop, hardware, and acquisition settings as the reactor runs. Continue each run until the detector signal returns to baseline. The apparatus response depends on flow rate, so the 500 µL/min response should not be reused for 300 or 700 µL/min without checking.

The next analysis should first fit and plot the blank reconstruction against the measured blank. Only if that fit reproduces the blank without oscillations or a large residual should we convolve the inferred response with the Gyroid CFD curve and call it a corrected comparison.

## New 500 µL/min convolution check (11 Nov 2026)

The supplied `rtd_curve (2).csv` was treated as the Gyroid CFD curve. Its calculated area is 1.000, mean 8.28 s, SD 7.97 s, skew 8.32, and peak 0.3023 s⁻¹ at 5.64 s.

For an exploratory correction, I built a causal gamma kernel matching the excess blank variance and third cumulant (external-response SD ≈17.0 s, skew ≈0.86; gamma shape ≈5.37, scale ≈7.3 s). After convolution, the predicted Gyroid curve and both experimental formulation means were normalized to unit area and aligned at detected rise.

The blank reconvolution has RMSE 13.7% of the measured blank peak. The validation plot shows a noticeably later predicted blank peak and a slower initial rise, so this is a useful sensitivity comparison but not a validated equipment correction. Absolute transit delays remain unavailable because these curves were independently aligned by detected rise.
