# HypoCover Results

Qwen2.5-0.5B-Instruct, PyTorch 2.15.0+cpu, Transformers 5.24.1, float32 on CPU. Both experiments use a fixed number of proposals per episode.

## Format-controlled experiment

On 18 additional projections, both conditions restrict the LLM to a height digit at each position and stop after the final column. The same public observation supplies the number of columns and maximum height. Neither condition receives a textual list of earlier proposals. The only treatment difference is the archive-based prefix tilt.

| Method | Validity | Uniqueness | Recovery | Distinct valid | Tokens per observation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Format-constrained sampling | 100.00% | 37.50% | 19.00% | 4.50 | 1392 |
| Format-constrained HypoCover | 100.00% | 86.11% | 50.62% | 10.33 | 1392 |

| Comparison | Change in Recovery, pp | Paired 95% interval, pp | Wins / ties / losses |
| --- | ---: | ---: | ---: |
| Format-constrained HypoCover vs. Format-constrained sampling | +31.62 | [+20.71, +43.90] | 18 / 0 / 0 |

| Legal structures | Observations | Format-constrained sampling | Format-constrained HypoCover |
| ---: | ---: | ---: | ---: |
| 9 | 6 | 33.33% | 98.15% |
| 27 | 6 | 16.05% | 39.51% |
| 81 | 6 | 7.61% | 14.20% |

The maximum macro-averaged Recovery possible under this proposal budget is 53.09%. It comes from taking the smaller of the proposal count and the legal solution count for each projection, then dividing by the number of legal solutions and averaging over projections.

## Free-form experiment

On a separate set of 18 projections, all seven methods generate without an output grammar. This tests whether the intervention helps when the model can still produce an invalid, malformed or cue-violating answer.

| Method | Validity | Uniqueness | Recovery | Distinct valid | Tokens per observation |
| --- | ---: | ---: | ---: | ---: | ---: |
| IID sampling | 73.15% | 43.06% | 17.42% | 4.33 | 1415 |
| Memory | 27.78% | 25.46% | 9.19% | 2.00 | 1630 |
| Static cues | 32.41% | 27.78% | 6.72% | 2.11 | 1719 |
| Oracle complexity | 19.44% | 27.31% | 8.50% | 1.56 | 1782 |
| Bandit only | 29.63% | 24.54% | 8.98% | 2.17 | 1704 |
| Tilt only | 19.91% | 26.85% | 8.16% | 2.28 | 1634 |
| HypoCover with prompt feedback | 29.63% | 34.72% | 13.85% | 3.33 | 1737 |

| Comparison | Change in Recovery, pp | Paired 95% interval, pp | Wins / ties / losses |
| --- | ---: | ---: | ---: |
| HypoCover with prompt feedback vs. Memory | +4.66 | [+0.00, +9.53] | 13 / 1 / 4 |
| HypoCover with prompt feedback vs. IID sampling | -3.57 | [-7.00, -0.27] | 3 / 3 / 12 |
| HypoCover with prompt feedback vs. Static cues | +7.13 | [+2.19, +12.69] | 13 / 2 / 3 |
| HypoCover with prompt feedback vs. Tilt only | +5.69 | [+1.37, +11.18] | 12 / 4 / 2 |
| HypoCover with prompt feedback vs. Bandit only | +4.87 | [+1.37, +8.64] | 13 / 3 / 2 |

| Legal structures | Observations | IID sampling | Memory | HypoCover with prompt feedback |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 6 | 29.63% | 18.52% | 25.93% |
| 27 | 6 | 14.81% | 5.56% | 9.26% |
| 81 | 6 | 7.82% | 3.50% | 6.38% |

## Interpretation

The format-controlled contrast isolates coverage feedback from validity because the grammar makes the same legal output space available to both methods. In the free-form experiment, cue adherence and answer formatting can trade off with exploration. The oracle complexity control uses the true voxel-count histogram to allocate prompts, while the model still has to follow each requested count.

Intervals use 5,000 paired bootstrap resamples at the observation level. These runs use one language model, one sampling seed, and one benchmark domain. Valid solutions can be directly constructed for this voxel task, so an unconstrained exact solver is a useful ceiling rather than an LLM baseline.

The JSON checkpoints contain every raw proposal and token count. The CSV files contain per-observation scores, and the summary JSON includes the full recovery curves.
