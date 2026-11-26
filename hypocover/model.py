"""Batched local Hugging Face inference. No network API is required."""

from collections import Counter
from dataclasses import dataclass
from math import log


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class PrefixFeedback:
    archive: tuple
    width: int
    alphabet: int
    strength: float
    grammar: bool = False


class CoverageTilt:
    """Tilt voxel digit logits by the unvisited fraction of each child prefix."""

    def __init__(self, plans, tokenizer, prompt_width, torch, device, stop_id):
        self.prompt_width = prompt_width
        self.stop_id = stop_id
        self.rows = []
        all_heights = max((plan.alphabet for plan in plans if plan), default=0)
        digit_ids = {
            height: tokenizer.encode(str(height), add_special_tokens=False)
            for height in range(1, all_heights + 1)
        }
        if any(len(ids) != 1 for ids in digit_ids.values()):
            raise ValueError("Coverage tilt requires one token for each height digit")
        reverse = {ids[0]: height for height, ids in digit_ids.items()}
        for row, plan in enumerate(plans):
            if not plan or (not plan.archive and not plan.grammar):
                continue
            prefixes = Counter(
                answer[:length] for answer in plan.archive for length in range(1, plan.width + 1)
            )
            ids = torch.tensor([digit_ids[h][0] for h in range(1, plan.alphabet + 1)], device=device)
            self.rows.append((row, plan, prefixes, ids))
        self.reverse = reverse

    def __call__(self, input_ids, scores):
        length = input_ids.shape[-1] - self.prompt_width
        prefixes = input_ids[:, self.prompt_width:].tolist() if length else None
        for row, plan, occupancy, ids in self.rows:
            if length >= plan.width:
                if plan.grammar:
                    scores[row].fill_(float("-inf"))
                    scores[row, self.stop_id] = 0
                continue
            prefix = tuple(self.reverse.get(token, -1) for token in prefixes[row]) if length else ()
            capacity = plan.alphabet ** (plan.width - length - 1)
            shifts = [
                plan.strength * log((capacity - occupancy[prefix + (h,)] + 0.5) / (capacity + 0.5))
                for h in range(1, plan.alphabet + 1)
            ]
            values = scores[row, ids] + scores.new_tensor(shifts)
            if plan.grammar:
                scores[row].fill_(float("-inf"))
            scores[row, ids] = values
        return scores


class LanguageModel:
    def __init__(self, name, device, dtype, threads, offline, batch_size, temperature, top_p):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")
        self.dtype = dtype if dtype != "auto" else ("bfloat16" if self.device == "cuda" else "float32")
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=offline, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=offline, dtype=getattr(torch, self.dtype)
        ).to(self.device).eval()
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.metadata = {
            "checkpoint": name,
            "revision": getattr(self.model.config, "_commit_hash", None),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": self.device,
            "hardware": torch.cuda.get_device_name() if self.device == "cuda" else "CPU",
            "dtype": self.dtype,
        }

    def generate(self, prompts, plans, seed, budget):
        from transformers import LogitsProcessorList

        torch = self.torch
        torch.manual_seed(seed)
        all_results = []
        for start in range(0, len(prompts), self.batch_size):
            texts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True
                )
                for prompt in prompts[start:start + self.batch_size]
            ]
            inputs = self.tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
            counts = inputs["attention_mask"].sum(dim=1).tolist()
            inputs = inputs.to(self.device)
            batch_plans = plans[start:start + self.batch_size]
            processors = LogitsProcessorList()
            if any(plan and (plan.archive or plan.grammar) for plan in batch_plans):
                processors.append(CoverageTilt(
                    batch_plans, self.tokenizer, inputs["input_ids"].shape[-1], torch, self.device,
                    self.tokenizer.eos_token_id,
                ))
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=budget, do_sample=True, temperature=self.temperature,
                    top_p=self.top_p, top_k=0, pad_token_id=self.tokenizer.pad_token_id,
                    logits_processor=processors,
                )
            generated = output[:, inputs["input_ids"].shape[-1]:].cpu().tolist()
            eos = self.model.generation_config.eos_token_id
            stop_ids = set(eos if isinstance(eos, list) else [eos])
            for ids, count in zip(generated, counts, strict=True):
                last = next((i + 1 for i, token in enumerate(ids) if token in stop_ids), len(ids))
                all_results.append(
                    Generation(self.tokenizer.decode(ids[:last], skip_special_tokens=True), count, last)
                )
        return all_results
