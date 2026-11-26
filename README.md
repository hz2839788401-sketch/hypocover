# HypoCover

HypoCover is a training-free decoding method for the gravity-constrained 3D reconstruction task in [HypoSpace](https://proceedings.mlr.press/v306/chen26bz.html). It uses previous valid proposals to shift token probability toward prefixes with unexplored completions. The repository compares this prefix penalty under a shared output grammar and studies its interaction with learned prompt routing in a separate free-form experiment. See [measured results and paired uncertainty intervals](RESULTS.md).

## Method

A valid reconstruction is uniquely represented by the bottom-contiguous heights of occupied columns in row-major order. For example, `123` represents heights 1, 2, and 3 at three observed occupied cells. Empty cells stay at height zero. We use this representation for every experimental condition.

1. **Prefix coverage tilt.** For each possible next height token, the decoder computes the fraction of completions of that digit prefix that have already been discovered. It lowers the token logit by the log of the smoothed unvisited fraction. This uses the known column-height bound and counts in a valid-output archive. The LLM samples each digit.
2. **Optional adaptive prompt routing.** A Thompson sampler chooses an unconstrained prompt or one of three height cues for a column selected by the run's seed. For each cue, it models the probability of a valid output and the conditional probability that a valid output is new. A discounted Beta update lets duplicate feedback gradually replace earlier novelty evidence while all validity evidence is retained.

If a child prefix has capacity `C` and contains `m` previously recovered candidates, the logit adjustment is `λ log((C − m + 0.5)/(C + 0.5))`. The capacity is `H ** remaining_columns`. Token logits outside the height alphabet are unchanged. Prompt cues are soft instructions; every proposal is scored against the original benchmark observation, even if the cue was missed.

With uniform digit logits, unit temperature, no top-p truncation, `λ = 1`, and zero smoothing, the rule reduces to uniform sampling over unseen legal completions. Every child at the same depth has equal capacity, so the conditional probability of a child is proportional to its number of unseen completions. These ratios telescope along a generated path, giving each unseen completion probability `1 / (total unseen completions)`. The experimental rule uses positive smoothing and the LLM's original logits, so this uniform special case does not describe the measured sampler.

## Evaluation

Each held-out set has 18 voxel observations, six each with 2, 3, and 4 observed columns. Each projection is on a 3×3 grid with maximum height 3. The corresponding solution counts are 9, 27, and 81. The free-form test, format-controlled test and six-case voxel pilot are disjoint. We matched the enumerated voxel solutions against the [official generator](https://github.com/CTT-Pavilion/_HypoSpace/blob/c69e9318577b34b5b896996571aefd4ba6053f58/3d/generate_3d_dataset_complete.py).

**Format-controlled comparison.** The baseline and the tilt both sample from legal digits for each occupied column, then stop. They share the same prompt, completion cap, temperature and top-p. They receive no textual history or cues. The grammar supplies the same legal output space for both methods. The logit penalty is the treatment.

**Free-form comparison.** These methods can emit malformed strings and cue-violating proposals. The table lists all seven conditions.

| Method | Prompt history | Prompt cue | Prefix tilt |
| --- | --- | --- | --- |
| IID | — | — | — |
| Memory | Last six distinct parseable proposals | — | — |
| Static | Same | Round-robin column-height cues | — |
| Oracle complexity | Same | Voxel-count cues sampled from the exact solution histogram | — |
| Bandit only | Same | Adaptive column-height cues | — |
| Tilt only | Same | — | Yes |
| HypoCover + routing | Same | Adaptive column-height cues | Yes |

The oracle comparator adapts the paper's complexity-stratified recipe to the voxel domain. It receives privileged histogram information, and it can miss the requested count. The primary baseline is memory prompting. The round-robin control uses the same cue pool as the bandit. All methods use the same model, temperature, top-p, number of proposals and maximum completion length. The evaluator also reports realized input and output token usage.

Validity is the fraction of attempts that satisfy the original observation. Uniqueness is the fraction of attempts producing a new parseable structure, including invalid structures. Recovery is the fraction of admissible structures recovered. Parse failures still consume one attempt. We macro-average over observations and use paired bootstrap intervals over observations. The format and fixed proposal count differ from the published leaderboard tables.

## Reproduce

Python 3.10+ and a local or downloadable Qwen2.5 checkpoint are required. CUDA is optional. Use a CUDA-enabled PyTorch build on a GPU machine.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m hypocover.data --per-stratum 6
python -m hypocover.run --tasks data/test.json --oracle-key data/test-key.json --out results/reproduction --domains voxel --methods iid memory static oracle bandit tilt adaptive --budget 12 --repeats 1 --batch-size 64 --threads 10 --max-new-tokens 7
python -m hypocover.evaluate --run results/reproduction --key data/test-key.json --out results/reproduction
python -m hypocover.run --tasks data/format-test.json --out results/grammar-reproduction --domains voxel --methods grammar grammar_tilt --budget 12 --repeats 1 --batch-size 36 --threads 6 --max-new-tokens 5
python -m hypocover.evaluate --run results/grammar-reproduction --key data/format-test-key.json --out results/grammar-reproduction
```

For cached checkpoints, add `--offline` to the run command. Resume an interrupted run by reusing its options and output path. The runner writes one atomic checkpoint per proposal round and verifies the replay when resuming. Counting tokens and reproducing the exact sample sequence can vary across PyTorch versions and hardware. The recorded environment and settings are in each run's `protocol.json`.

To independently check the answer keys, run `python tools/verify_upstream.py /path/to/_HypoSpace` with the authors' repository at commit `c69e9318577b34b5b896996571aefd4ba6053f58`.

The answer key is used by offline evaluation and by the explicitly privileged oracle policy. No other policy is passed the list of answers or the complexity histogram. Use the run command without `--oracle-key` and omit `oracle` from `--methods` for an entirely key-free generation process.

## Scope

For this factorized voxel task, direct enumeration or sampling without replacement is an exact algorithmic reference. We study what changes in the LLM proposal distribution when an archive informs decoding. Generalization across domains, model sizes, and random seeds requires additional evaluation. The causal task adapter is exploratory and outside the reported experiment.

## References

- Chen et al. (2026). [HypoSpace](https://proceedings.mlr.press/v306/chen26bz.html). ICML. See §5.2 for the voxel task and §6.5 for complexity-stratified decoding.
- Wong et al. (2025). [SimpleStrat](https://papers.neurips.cc/paper_files/paper/2025/file/a99cbe298888dc87029faa80eb10d2b9-Paper-Conference.pdf). NeurIPS. Prior work on partition-based diversification of LLM outputs.
- Kool et al. (2019). [Stochastic Beams and Where To Find Them](https://proceedings.mlr.press/v97/kool19a.html). ICML. Exact sequence sampling without replacement provides a related approach to avoiding duplicates. The prefix tilt here uses partial occupancy of legal completions and still permits repeats.
