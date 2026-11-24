"""Two-stage disk capture through the existing MiaLLM API on a real GPU."""
import json

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")
from vllm import SamplingParams

from mia import MiaLLM
from mia.artifacts import load_and_merge_hs_cache, unpack_hidden_states
from tests.conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("safetensors", [False, True])
def test_two_stage_disk_capture(tmp_path, monkeypatch, engines, enforce_eager, safetensors):
    """Chunked prompts and unequal decode lengths retain their order in both formats."""
    monkeypatch.setenv("MIA_USE_SAFETENSORS", str(int(safetensors)))
    monkeypatch.setenv("MIA_APERTURE_GPU_BYTES", str(128 << 20))
    config = tmp_path / "infer.json"
    config.write_text(json.dumps({"hidden_states": {"layers": [2, 3], "mode": "last_token"}}))
    hook_dir = str(tmp_path / "hooks")
    llm = MiaLLM(model="facebook/opt-125m", worker_name="capture_hs",
                 analyzer_name="two_stage_probe", config_file=str(config), hook_dir=hook_dir,
                 enforce_eager=enforce_eager, dtype="float16", gpu_memory_utilization=0.2,
                 enable_prefix_caching=False, max_model_len=1024,
                 max_num_batched_tokens=128, max_num_seqs=3)
    engines.append(llm)

    # Each probe reads a different layer; scores are checked without the analyzer's helpers.
    hidden = llm.llm.llm_engine.model_config.hf_config.hidden_size
    rng = np.random.default_rng(4)
    weights = rng.normal(0, 0.02, size=hidden).astype(np.float32)
    paths = []
    for layer in (2, 3):
        path = tmp_path / str(layer)
        path.mkdir()
        np.savez(path / "probe.npz", mean=np.zeros(hidden, np.float32),
                 scale=np.ones(hidden, np.float32), weights=weights,
                 bias=np.float32(-0.1), layer=np.int64(layer))
        (path / "probe.json").write_text(json.dumps(
            {"model_name": "facebook/opt-125m", "layer": layer, "threshold": 0.5}))
        paths.append(str(path / "probe.npz"))

    prompts = ["The capital of France is", "A small blue bird sings. " * 40,
               "Count from one to ten:"]
    lengths = [9, 5, 7]  # completion order differs from input order
    sampling = [SamplingParams(temperature=0, max_tokens=n, ignore_eos=True,
                               extra_args={"hooks_on": "both"}) for n in lengths]
    outputs = llm.generate(prompts, sampling, save_to_disk=True, run_id="decode")
    assert [len(o.outputs[0].token_ids) for o in outputs] == lengths
    assert len(outputs[1].prompt_token_ids) > 128
    capture = load_and_merge_hs_cache(hook_dir, "decode")
    by_layer = {}
    for entry in capture["hs_cache"].values():
        assert entry["req_passes"] == lengths
        by_layer[entry["layer_num"]] = list(torch.split(
            torch.stack(unpack_hidden_states(entry)).float(), lengths))

    # Independent single-request captures detect a batch/request alignment error.
    for i, (prompt, sp) in enumerate(zip(prompts, sampling)):
        one = llm.generate([prompt], sp)[0].probes
        for entry in one["hs_cache"].values():
            rows = torch.stack(unpack_hidden_states(entry)).reshape(lengths[i], hidden).float()
            expected = by_layer[entry["layer_num"]][i]
            assert torch.nn.functional.cosine_similarity(rows, expected, dim=-1).min() > 0.999

    windows = [[1, 4], None, [2, 20]]
    def probability(rows):
        return 1 / (1 + np.exp(-(rows.numpy() @ weights - 0.1)))

    for position in ("prompt_last", "decode_mean", "decode_window"):
        spec = {"stage1": {"probe_path": paths[0], "threshold": 0.5},
                "stage2": {"product": [
                    {"probe_path": paths[0]},
                    {"probe_path": paths[1], "position": position, "decode_tokens": 3}],
                    "threshold": 0.25}, "score_all": True, "decode_windows": windows}
        result = llm.analyze(analyzer_spec=spec, run_id="decode")
        want1, want2 = [], []
        for i in range(len(prompts)):
            first, second = by_layer[2][i], by_layer[3][i]
            if position == "prompt_last":
                feature = second[0]
            else:
                start, end = windows[i] if position == "decode_window" and windows[i] else (0, 3)
                feature = second[1 + start:1 + end].mean(0)
            want1.append(probability(first[0]))
            want2.append(probability(first[0]) * probability(feature))
        np.testing.assert_allclose(result["stage1"]["probabilities"], want1, atol=1e-6)
        np.testing.assert_allclose(result["stage2"]["probabilities"], want2, atol=1e-6)
        assert result["verdicts"] == ["flag" if a >= 0.5 and b >= 0.25 else "pass"
                                      for a, b in zip(want1, want2)]
        spec["score_all"] = False
        gated = llm.analyze(analyzer_spec=spec, run_id="decode")
        assert gated["verdicts"] == result["verdicts"]
        assert [p is not None for p in gated["stage2"]["probabilities"]] == [a >= 0.5 for a in want1]

    prompt_spec = {"stage1": {"probe_path": paths[0], "threshold": 0.5},
                   "stage2": {"probe_path": paths[1], "threshold": 0.5}, "score_all": True}
    sp = SamplingParams(temperature=0, max_tokens=1, extra_args={"hooks_on": "prefill"})
    llm.generate(prompts, sp, save_to_disk=True, run_id="prompt")
    disk = llm.analyze(analyzer_spec=prompt_spec, run_id="prompt")
    merged = llm.generate(prompts, sp)[0].probes
    memory = llm.analyze(analyzer_spec=prompt_spec, probes=merged)
    for stage in ("stage1", "stage2"):
        np.testing.assert_allclose(disk[stage]["probabilities"], memory[stage]["probabilities"], atol=1e-5)
    assert disk["verdicts"] == memory["verdicts"]
