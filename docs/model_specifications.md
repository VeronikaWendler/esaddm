# Systems

All value inputs are objective probabilities on a 0–1 scale. Values and
gaze fractions are rounded to three decimals, and individual contribution terms
are rounded before assembling regressors.

## Accuracy-coded ES / EE

`response = 1` if the higher-valued physical option was chosen; otherwise 0.
After three-decimal probability rounding, equal-value trials are removed otherwise this code doesn't work.
RT >= 0.250 seconds (Ting & Gluth, 2025).

Let `g_high`, `g_low` be dwell fractions for higher-/lower-valued options:

- `attention_w = round(g_high * v_high - g_low * v_low, 3)`
- `inattention_w = round(g_low * v_high - g_high * v_low, 3)`

drift model: `v ~ 0 + attention_w + inattention_w`.


## Identity-coded ES (primary four-model family)

`response = 1` if S was chosen; `response = 0` if E was chosen regardless of side.
Equal-value trials are included. RT must be at least 0.250 seconds.

For `g_s`, `g_e`, values `v_s`, `v_e`:

- `attended_s = round(g_s * v_s, 3)`
- `attended_e = round(-g_e * v_e, 3)`
- `unattended_s = round(g_e * v_s, 3)`
- `unattended_e = round(-g_s * v_e, 3)`
- `attention_w = round(attended_s + attended_e, 3)`
- `inattention_w = round(unattended_s + unattended_e, 3)`
- `attention_contrast = round((attended_s - attended_e) / 2, 3)`
- `inattention_contrast = round((unattended_s - unattended_e) / 2, 3)`

**Four intended fits** (all with `v ~ 0 + ...`):

| Model | Drift predictors | Starting point |
| --- | --- | --- |
| `addm` | `attention_w`, `inattention_w` | z fixed 0.5 |
| `addm_z` | same | z estimated |
| `esaddm` | base + `attention_contrast`, `inattention_contrast` | z fixed 0.5 |
| `esaddm_z` | same contrast features | z estimated |

A coefficient on the signed contrast has interpretation `delta_a = b_as-b_ae`
(and `delta_i = b_is-b_ie`), because the contrast is half the difference.


## Inclusions and expected samples

| | Study 1 | Study 2 |
| --- | --- | --- |
| Legacy exclusions | 1,4,5,6,14,99 | 2,6,9,14,18,20,26 |
| Participants | 21 | 20 |
| Identity ES rows | 7,234 | 6,549 |
| Accuracy ES rows | 6,435 | 5,811 |
| Accuracy EE rows | 2,756 | 2,363 |