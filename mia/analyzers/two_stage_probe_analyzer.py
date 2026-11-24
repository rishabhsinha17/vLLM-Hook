"""Two-stage probe analyzer: a stage-1 probe gates, a stage-2 scorer confirms.

Every probe is a linear probe in the ``tool_call_risk`` artifact format (``probe.npz`` +
``probe.json``). A step is flagged only when both stages fire, and stage 2 is scored only on
the steps stage 1 flags. A stage is either one probe or ``{"product": [probe, ...]}``, whose
score is the product of its probes' probabilities (all of them must agree).

Each probe reads one layer at one position:
  prompt_last   hidden state of the last prompt token (default; works on merged probes)
  decode_mean   mean of the states for generated tokens 1..decode_tokens. This needs each
                request's prompt and decode passes: use ``hooks_on="both"``,
                ``hidden_states.mode="last_token"`` and ``save_to_disk=True``, then
                analyze by ``run_id``. The artifact's ``req_passes`` separates requests.
  decode_window mean of the states for the generated tokens in a per-request window, given as
                ``"decode_windows": [[start, end] or None, ...]`` in the spec, one entry per
                request (0-based generated-token indices, end exclusive; the spec must hold it).
                A request whose entry is None falls back to the first decode_tokens tokens, as
                decode_mean. ``mia.utils.two_stage.tool_call_window`` finds the window of the
                first tool call. Same capture as decode_mean.
``position`` and ``decode_tokens`` default to the probe's probe.json and can be set in the spec.
Nothing here depends on the task the probes were trained for.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from mia.artifacts import load_and_merge_hs_cache, unpack_hidden_states
from mia.utils.tool_call_risk.score import ToolCallRiskProbe
from mia.utils.two_stage.cascade import TwoStageDetector

STAGES = ("stage1", "stage2")
POSITIONS = ("prompt_last", "decode_mean", "decode_window")


def _layer_entry(hs_cache: Dict, layer: int) -> Dict:
    for entry in hs_cache.values():
        if int(entry["layer_num"]) == layer:
            return entry
    available = sorted(int(e["layer_num"]) for e in hs_cache.values())
    raise RuntimeError(
        f"A probe expects activations from layer {layer}, but the current config captured "
        f"layers {available}. Add layer {layer} to 'hidden_states.layers'."
    )


def last_token_features(hs_cache: Dict, layer: int) -> np.ndarray:
    """(steps, hidden) last-prompt-token states at ``layer`` from a merged capture."""
    tensors: List[torch.Tensor] = unpack_hidden_states(_layer_entry(hs_cache, layer))
    rows = [t if t.dim() == 1 else t[-1] for t in tensors]
    return torch.stack(rows).float().cpu().numpy()


def _request_entries(entry: Dict) -> List[Dict]:
    """Split a disk capture by the existing per-request pass counts.

    Last-token capture keeps only the final prefill pass, even with chunked prefill.
    All-token disk captures do not retain the prefill/decode boundary for each request.
    """
    passes = unpack_hidden_states(entry)
    counts = entry["req_passes"]
    if (any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts)
            or sum(counts) != len(passes)):
        raise ValueError("req_passes must contain nonnegative integers summing to the number of passes")
    if not counts:
        raise ValueError("capture contains no requests")
    if entry.get("hs_mode", "last_token") != "last_token" and any(n > 1 for n in counts):
        raise ValueError("Grouped captures with multiple passes need hidden_states.mode='last_token'; "
                         "all_tokens disk artifacts do not retain each request's prefill boundary")
    requests, start = [], 0
    for i, n in enumerate(counts):
        if not n:
            raise ValueError(f"No hidden states for request {i}; capture every request before scoring")
        one = {**entry, "hidden_states": passes[start:start + n]}
        one.pop("req_passes")
        if entry.get("pass_lens") is not None:
            one["pass_lens"] = entry["pass_lens"][start:start + n]
        requests.append(one)
        start += n
    return requests


def request_features(probes: Dict, layer: int, position: str = "prompt_last",
                     decode_tokens: int = 16, window: Optional[Sequence[int]] = None) -> np.ndarray:
    """(hidden,) feature of one request from its own capture.

    Decode pass ``t`` (0-based) is the state whose input is generated token ``t``, so the last
    generated token has no state. ``window`` is used by ``decode_window`` only.
    """
    entry = _layer_entry(probes["hs_cache"], layer)
    tensors = unpack_hidden_states(entry)
    lens = entry.get("pass_lens")
    if lens is not None and len(lens) != len(tensors):
        raise ValueError("pass_lens must have one length per captured pass")
    passes = []
    for i, t in enumerate(tensors):
        if t is None or not t.numel():
            raise ValueError("request contains an empty hidden-state pass")
        n = int(lens[i]) if lens is not None else (len(t) if t.dim() > 1 else 1)
        if not 0 < n <= (len(t) if t.dim() > 1 else 1):
            raise ValueError("pass_lens must identify real rows within each captured pass")
        passes.append(t if t.dim() == 1 else t[n - 1])
    n_prefill = int(entry.get("prefill_passes", entry.get("_n_prefill", 1)))
    if not 0 < n_prefill <= len(passes):
        raise ValueError("request needs a captured prompt pass; use hooks_on='both' for decode probes")
    if position == "prompt_last":
        return passes[n_prefill - 1].float().cpu().numpy()
    if position not in ("decode_mean", "decode_window"):
        raise ValueError(f"position must be one of {POSITIONS}, got {position!r}")
    if decode_tokens <= 0:
        raise ValueError("decode_tokens must be positive")
    start, end = 0, decode_tokens
    if position == "decode_window" and window is not None:
        start, end = int(window[0]), int(window[1])
        if not 0 <= start < end:
            raise ValueError(f"decode window must satisfy 0 <= start < end, got {window!r}")
    dec = passes[n_prefill + start:n_prefill + end]
    if not dec:
        raise ValueError(
            f"{position} needs decode-pass states for generated tokens {start}..{end - 1}: generate with "
            "SamplingParams(extra_args={'hooks_on': 'both'}) and enough max_tokens")
    return torch.stack(dec).float().mean(0).cpu().numpy()


def _resolve_probe(spec: Dict, base: Optional[str]) -> Dict:
    spec = dict(spec)
    p = spec.get("probe_path")
    if not p:
        raise ValueError("every probe in a stage needs a 'probe_path'")
    if base and not os.path.isabs(p):
        spec["probe_path"] = os.path.join(base, p)
    return spec


def resolve_spec(spec: Dict) -> Dict[str, Dict]:
    """Merge ``cascade_path`` (if given) with per-stage overrides from the spec.

    ``cascade.json`` holds {"stage1": {...}, "stage2": {...}}, each stage either
    {"probe_path", "threshold", ...} or {"product": [{"probe_path", ...}, ...], "threshold"}.
    Relative probe paths are resolved against the directory of cascade.json.
    """
    stages: Dict[str, Dict] = {s: {} for s in STAGES}
    base = None
    path = spec.get("cascade_path")
    if path:
        with open(path) as f:
            cfg = json.load(f)
        base = os.path.dirname(os.path.abspath(path))
        for s in STAGES:
            stages[s].update(cfg.get(s, {}))
    for s in STAGES:
        stages[s].update(spec.get(s, {}))
        members = stages[s].get("product")
        if members is None:
            if not stages[s].get("probe_path"):
                raise ValueError(
                    "TwoStageProbeAnalyzer needs analyzer_spec={'cascade_path': ...} or "
                    f"{{'stage1': {{'probe_path': ...}}, 'stage2': {{'probe_path': ...}}}} (missing {s})"
                )
            members = [{k: v for k, v in stages[s].items() if k != "threshold"}]
        stages[s]["members"] = [_resolve_probe(m, base) for m in members]
    return stages


class TwoStageProbeAnalyzer:
    """Flags a step when the stage-1 and stage-2 scores both clear their thresholds."""

    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        """``hook_dir`` holds the disk runs; ``layer_to_heads`` is unused."""
        self.hook_dir = hook_dir
        self._probes: Dict[str, ToolCallRiskProbe] = {}

    def _probe(self, path: str) -> ToolCallRiskProbe:
        if path not in self._probes:
            self._probes[path] = ToolCallRiskProbe.load(path)
        return self._probes[path]

    def _features(self, member: Dict, probe: ToolCallRiskProbe,
                  probes: Union[Dict, Sequence[Dict]], windows: Optional[Sequence] = None) -> np.ndarray:
        position = member.get("position", probe.artifact.meta.get("position", "prompt_last"))
        k = int(member.get("decode_tokens", probe.artifact.meta.get("decode_tokens", 16)))
        if isinstance(probes, dict):
            entry = _layer_entry(probes["hs_cache"], probe.layer)
            if "req_passes" not in entry:
                if position != "prompt_last":
                    raise ValueError(
                        f"{member['probe_path']} reads position {position!r}, which needs request "
                        "boundaries. Generate with save_to_disk=True and hooks_on='both', then "
                        "analyze by run_id; alternatively pass a list of single-request captures.")
                return last_token_features(probes["hs_cache"], probe.layer)
            probes = [{"hs_cache": {"layer": e}} for e in _request_entries(entry)]
        if position == "decode_window" and windows is None:
            raise ValueError(
                f"{member['probe_path']} reads position 'decode_window', so the spec needs "
                "'decode_windows': one [start, end] or None per request, for example from "
                "mia.utils.two_stage.tool_call_window(tokenizer, output.token_ids).")
        if windows is not None and len(windows) != len(probes):
            raise ValueError(f"decode_windows has {len(windows)} entries for {len(probes)} requests")
        return np.stack([request_features(p, probe.layer, position, k, windows[i] if windows is not None else None)
                         for i, p in enumerate(probes)])

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        run_id: Optional[str] = None,
        probes: Optional[Union[Dict, Sequence[Dict]]] = None,
    ) -> Dict:
        """Score each step with both stages.

        ``run_id`` loads a disk capture, including decode passes grouped by ``req_passes``.
        ``probes`` can also be a prompt-only merged capture (``out[0].probes``), a loaded
        disk capture, or a list of captures from separate single-request calls. Both
        thresholds must come from ``cascade.json`` or the spec; probe.json thresholds are
        ignored because each was calibrated for its probe alone. ``"score_all": true``
        scores stage 2 on every step, which recalibration needs.
        """
        spec = analyzer_spec or {}
        stages = resolve_spec(spec)
        missing = [s for s in STAGES if stages[s].get("threshold") is None]
        if missing:
            raise ValueError(
                f"TwoStageProbeAnalyzer needs a calibrated threshold for {', '.join(missing)}. "
                "A probe.json threshold was set for that probe alone, so it is not used here. "
                "Calibrate both with mia.utils.two_stage.calibrate_cascade and write them with "
                "save_cascade_config, or pass them in the spec."
            )
        t1, t2 = float(stages["stage1"]["threshold"]), float(stages["stage2"]["threshold"])

        if probes is None:
            if run_id is None:
                raise ValueError("TwoStageProbeAnalyzer.analyze: pass either probes= or run_id=.")
            probes = load_and_merge_hs_cache(self.hook_dir, run_id)
        elif isinstance(probes, dict) and "hs_cache" not in probes:
            raise ValueError("probes must hold 'hs_cache' (a capture_hs run)")

        feats = {}
        for s in STAGES:
            feats[s] = []
            for m in stages[s]["members"]:
                pr = self._probe(m["probe_path"])
                feats[s].append((pr, self._features(m, pr, probes, spec.get("decode_windows"))))
        n = len(feats["stage1"][0][1])
        if any(len(x) != n for stage in feats.values() for _, x in stage):
            raise ValueError("Every probe layer must contain the same number of requests")

        def scorer(stage):
            def run(idx):
                out = np.ones(len(idx))
                for pr, x in feats[stage]:
                    out *= np.array([sc.probability for sc in pr.score(x[idx])])
                return out
            return run

        result = TwoStageDetector(scorer("stage1"), scorer("stage2"), t1, t2)(
            np.arange(n), score_all=bool(spec.get("score_all", False)))

        def layers(stage):
            ls = [pr.layer for pr, _ in feats[stage]]
            return ls[0] if len(ls) == 1 else ls

        return {
            "verdicts": ["flag" if f else "pass" for f in result.flags],
            "reached_stage2": [bool(r) for r in result.reached_stage2],
            "stage1": {
                "probabilities": result.stage1_scores.tolist(),
                "layer": layers("stage1"),
                "threshold": t1,
            },
            "stage2": {
                "probabilities": [None if np.isnan(p) else float(p) for p in result.stage2_scores],
                "layer": layers("stage2"),
                "threshold": t2,
            },
        }
