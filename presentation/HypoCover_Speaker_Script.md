# HypoCover 10 Minute Presentation Script

Three speakers covering the task, the method, and the experiments.

Times are recording marks. Bracketed cues are on-screen actions. Speak at about 120–130 words per minute, with pauses at the diagrams.

Speaker A rehearses 0:00–3:15, B 3:15–6:45, and C 6:45–10:00.

## Speaker A

### 0:00–0:55  |  Slide 1  |  Finding more valid explanations

*[Show the title]*

Given an observation, we often have several explanations that all fit. Imagine a camera looking straight down at blocks on a table. It can tell us which positions contain blocks. It cannot tell us how many blocks are stacked at each position. If we ask a language model for twelve plausible reconstructions and it gives the same answer twelve times, every answer might be valid. We would still have only one explanation. That is the behavior we study with HypoCover. Our question is whether an LLM can find more distinct valid hypotheses if every new proposal is informed by the part of the space it has already visited. We use the voxel task from the ICML HypoSpace benchmark to measure that directly.

### 0:55–2:00  |  Slide 2  |  One view, many reconstructions

*[Point to the shared top view]*

Look at the three green squares in the top view. They tell us that three columns are occupied. Gravity means that a block cannot float above an empty layer. So, once we know a column's height, we know exactly which blocks are in it. Both reconstructions on the right satisfy the same observation. With a maximum height of three, each marked square has three possible heights. For three marked squares, that gives twenty-seven possible structures. We encode each answer as a string of three digits, in row-major order. That representation preserves the full set of valid reconstructions and lets us identify duplicates exactly, even when a model varies the whitespace. We checked every selected projection against the authors' generator. Both the baselines and our method work with the same observations and the same answer format.

### 2:00–3:15  |  Slide 3  |  How we measure exploration

*[Point to the repeated answer]*

We report three metrics. Validity asks how many proposals satisfy the original top view and height constraints. Uniqueness asks how many proposals are different, including parseable proposals that are invalid. Recovery asks how many members of the valid set the model actually found. In the example on this slide, the model produces one-two-one, repeats it, then proposes three-one-two. Every answer is valid. Two-thirds of the proposals are unique, and the run covers two of the twenty-seven possible structures. Recovery is our primary metric because it rewards validity and discovery together. The benchmark paper already explores complexity-stratified prompts. We keep fixed prompts and memory prompts as controls. We also include a complexity baseline that receives the true size of each complexity group. Our new intervention uses the archive during token generation as well as between proposals. Speaker B will show how those two feedback loops work.

## Speaker B

### 3:15–4:20  |  Slide 4  |  Two feedback loops

*[Follow the arrows from prompt to archive]*

Thanks. The generator is a frozen instruction-tuned model. The system supplies the top view, a short list of previous proposals, and optionally a cue such as 'column B must have height three.' The model produces one candidate. A deterministic validator checks that candidate, and an archive records distinct valid answers. The archive informs two decisions. Between proposals, the policy learns which cues tend to lead to fresh valid answers. During a proposal, the decoder checks whether a digit would continue a prefix already represented in the archive. The budget charges one attempt for each model output, including duplicates and malformed strings. The archive contains model proposals only. We can compute how many structures could extend a given prefix from the remaining column count and the maximum height. That information lets us estimate how crowded a prefix is without giving the generator the answers.

### 4:20–5:35  |  Slide 5  |  Reweighting the next height

*[Point to prefix 12]*

Here is the token-level rule. Suppose the first generated height is one. We are choosing the second height in a three-column reconstruction. If we choose two, the prefix becomes one-two. Three valid structures have that prefix, since the last column can take any of three heights. If two have already been found, the branch has one unseen completion. The decoder lowers the logit for the token two by a soft penalty based on that fraction. If the prefix one-three has no entries in the archive, its logit stays unchanged. The model still assigns the starting scores to both digits. The small smoothing constant keeps the penalty finite even when a branch is full. We chose a soft penalty because the parser and validator should still measure malformed or incorrect proposals. The intervention redistributes probability within the model's generation process, and every output still goes through the same evaluation.

### 5:35–6:45  |  Slide 6  |  Learning where to spend proposals

*[Compare the two ablations]*

At the proposal level, consider two prompts. One gives a valid structure almost every time but keeps repeating it. Another has slightly lower validity and finds a new structure when it works. We estimate two probabilities for each prompt: validity, and freshness conditional on validity. Their product estimates the chance of a new valid discovery. Thompson sampling draws both probabilities from their current distributions and chooses the cue with the highest sampled product. We keep the validity evidence across the run and discount older freshness evidence because novelty wears out as the archive grows. Fixed routing cycles through the same cue choices. The bandit-only ablation uses the learned prompts without the token penalty, while tilt only keeps the memory prompt and applies the token penalty. Those two ablations let us ask which feedback signal is driving the result. Speaker C will take us through the experiment.

## Speaker C

### 6:45–7:55  |  Slide 7  |  Recovery with the same output grammar

*[Trace both recovery curves]*

Here is the format-controlled test. We evaluated Qwen two-point-five, zero-point-five billion parameters, on 18 independent voxel projections with 12 queries each. Both conditions restrict the output to legal height digits and use the same prompt. As a result, both are valid by construction. The prefix penalty is the only thing that changes. Ordinary sampling recovered 19.0 percent of the valid structures on average. HypoCover recovered 50.6 percent, a change of +31.6 percentage points. The paired bootstrap interval is +20.7 to +43.9 points. The interval excludes zero in this run. It improved on 18 of the 18 projections. Since the grammar, prompt and actual token counts are shared, this comparison isolates the effect of archive-based coverage feedback. The budget caps the best possible average Recovery at about fifty-three percent across these three difficulty levels.

### 7:55–9:05  |  Slide 8  |  The free-form experiment

*[Compare the ablation rows]*

We also ran a free-form experiment on a separate set of 18 projections. Here, the LLM can output invalid heights or fail to follow the format. Independent sampling recovered 17.4 percent, memory prompting recovered 9.2 percent, and the combined method recovered 13.9 percent. Bandit only reached 9.0 percent and tilt only reached 8.2 percent. Combining these components yielded higher Recovery than either component alone in this free-form comparison. The oracle complexity control receives the true histogram of voxel counts. Its result also depends on whether the model follows the requested count. We report actual tokens per case because longer memory prompts change realized cost even when the number of proposals and completion cap are held fixed.

### 9:05–10:00  |  Slide 9  |  What this result establishes

*[Show the reproducibility checklist]*

We evaluated one small model and one domain. The voxel task is useful because we can count valid completions exactly. A classical generator can enumerate these structures outright, so our research claim concerns the LLM's sampling distribution on a controlled benchmark. For tasks such as causal graph generation, the same prefix-capacity calculation may be unavailable or expensive. Those tasks would need a different capacity estimate and their own evaluation. The project repository includes the dataset generator, the inference policies, the raw proposal traces, and the evaluator. You can reproduce the curves from the traces without rerunning the language model. Our takeaway is that the record of what an LLM has already found can be used inside decoding. That gives us a way to measure and target missed coverage under a controlled generation budget. Thank you.
