from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

RUNTIME_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as exc:
    torch = None
    ProfilerActivity = None
    profile = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    RUNTIME_IMPORT_ERROR = exc


DEFAULT_PROMPT = "In one sentence, explain why byte-level language models are useful:"
DEFAULT_CONTINUATION = (
    " they can model raw text without a fixed tokenizer while spending compute where the"
    " byte stream is harder to predict."
)


def default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Measurement:
    flops: int
    elapsed_seconds: float
    peak_cuda_memory_bytes: int | None


@dataclass
class ModelResult:
    name: str
    model_id: str
    backend: str
    mode_requested: str
    mode_used: str
    dtype: str
    device: str
    prompt_tokens: int
    continuation_tokens: int
    continuation_bytes: int
    flops: int
    elapsed_seconds: float
    peak_cuda_memory_bytes: int | None
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "compare profiler-counted flops for blt and a tokenized 1b model while"
            " generating the same fixed continuation text"
        )
    )
    parser.add_argument("--blt-model", default="itazap/blt-1b-hf")
    parser.add_argument(
        "--blt-backend",
        choices=["auto", "transformers", "native"],
        default="auto",
        help=(
            "auto uses transformers when installed support exists and falls back to this"
            " repo's native blt loader otherwise"
        ),
    )
    parser.add_argument(
        "--native-blt-model",
        default="facebook/blt-1b",
        help="hub repo used when --blt-backend native or auto fallback is selected",
    )
    parser.add_argument(
        "--native-blt-entropy-model",
        default="facebook/blt-entropy",
        help="entropy model repo used by the native blt patcher",
    )
    parser.add_argument("--llama-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--continuation", default=DEFAULT_CONTINUATION)
    parser.add_argument(
        "--mode",
        choices=["cached", "nocache"],
        default="cached",
        help=(
            "cached profiles the normal kv-cache generation path; nocache recomputes the"
            " full prefix at every generated token"
        ),
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--device",
        default=default_device(),
        help="device for both models, for example cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="pass trust_remote_code=true to transformers loaders",
    )
    parser.add_argument(
        "--no-fallback-nocache",
        action="store_true",
        help="fail instead of rerunning nocache if cached mode is unsupported",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional path to write machine-readable results",
    )
    return parser.parse_args()


def choose_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def load_transformers_model_and_tokenizer(
    model_id: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool,
) -> tuple[torch.nn.Module, object]:
    token = os.environ.get("HF_TOKEN")
    auth_kwargs = {"token": token} if token else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        **auth_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        **auth_kwargs,
    )
    model.eval()
    model.to(device)
    return model, tokenizer


def hf_token_kwargs() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    return {"token": token} if token else {}


def download_hub_file(model_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(model_id, filename, **hf_token_kwargs())
    except Exception as exc:
        message = str(exc)
        if "gated repo" in message.lower() or "401" in message:
            raise RuntimeError(
                f"cannot access {model_id}/{filename}; set HF_TOKEN and make sure your"
                " hugging face account has access to the gated BLT weights"
            ) from exc
        raise


def load_hub_json(model_id: str, filename: str) -> dict[str, Any]:
    path = download_hub_file(model_id, filename)
    with open(path) as handle:
        return json.load(handle)


def extract_hub_args(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    if isinstance(config.get("args"), dict):
        return config["args"]
    if isinstance(config.get("config"), dict) and isinstance(
        config["config"].get("args"),
        dict,
    ):
        return config["config"]["args"]
    raise ValueError(
        f"could not find an args object in {model_id}/config.json; config keys are"
        f" {sorted(config.keys())}"
    )


def load_safetensors_state_dict(model_id: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    model_path = download_hub_file(model_id, "model.safetensors")
    return load_file(model_path, device="cpu")


def strip_state_dict_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not all(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key.removeprefix(prefix): value for key, value in state_dict.items()}


def load_state_dict_strictly(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    model_id: str,
) -> None:
    candidates = [
        state_dict,
        strip_state_dict_prefix(state_dict, "model."),
        strip_state_dict_prefix(state_dict, "_orig_mod."),
    ]
    errors: list[str] = []
    seen_key_sets: set[tuple[str, ...]] = set()
    for candidate in candidates:
        key_signature = tuple(candidate.keys())
        if key_signature in seen_key_sets:
            continue
        seen_key_sets.add(key_signature)
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except RuntimeError as exc:
            errors.append(str(exc).splitlines()[0])

    raise RuntimeError(
        f"failed to load {model_id} state dict strictly; attempted key layouts reported:"
        f" {errors}"
    )


def load_native_hub_model(
    model_id: str,
    model_cls: type[torch.nn.Module],
    args_cls: type[Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    config = load_hub_json(model_id, "config.json")
    args = args_cls(**extract_hub_args(config, model_id))
    model = model_cls(args)
    load_state_dict_strictly(model, load_safetensors_state_dict(model_id), model_id)
    model.eval()
    model.to(device=device, dtype=dtype)
    return model


def load_native_blt(
    blt_model_id: str,
    entropy_model_id: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.nn.Module, object, object]:
    from bytelatent.data.patcher import to_device
    from bytelatent.hf import BltTokenizerAndPatcher
    from bytelatent.model.blt import ByteLatentTransformer, ByteLatentTransformerArgs
    from bytelatent.transformer import LMTransformer, LMTransformerArgs

    entropy_model = load_native_hub_model(
        entropy_model_id,
        LMTransformer,
        LMTransformerArgs,
        dtype=dtype,
        device=device,
    )
    blt_model = load_native_hub_model(
        blt_model_id,
        ByteLatentTransformer,
        ByteLatentTransformerArgs,
        dtype=dtype,
        device=device,
    )
    tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(
        blt_model_id,
        **hf_token_kwargs(),
    )

    tokenizer = tok_and_patcher.tokenizer_args.build()
    patcher_args = tok_and_patcher.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = False
    patcher_args.device = str(device)
    patcher_args.patching_device = str(device)
    patcher = patcher_args.build()
    patcher.realtime_patching = True
    entropy_model, patching_device = to_device(entropy_model, str(device))
    patcher.entropy_model = entropy_model.eval()
    patcher.device = str(patching_device)

    patcher.entropy_model.to(device=device, dtype=dtype)
    return blt_model, tokenizer, patcher


def encode_text(
    tokenizer: object,
    text: str,
    device: torch.device,
    *,
    add_special_tokens: bool,
) -> torch.Tensor:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    if input_ids.numel() == 0:
        raise ValueError("text encoded to zero tokens")
    return input_ids


def encode_native_blt_text(
    tokenizer: object,
    text: str,
    device: torch.device,
    *,
    add_bos: bool,
) -> torch.Tensor:
    token_ids = tokenizer.encode(text, add_bos=add_bos, add_eos=False)
    if not token_ids:
        raise ValueError("text encoded to zero tokens")
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def run_cached_generation_loop(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
) -> None:
    with torch.inference_mode():
        output = model(input_ids=prompt_ids, use_cache=True)
        past_key_values = getattr(output, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError("model did not return past_key_values for cached decoding")

        # generating n tokens needs the prompt prefill plus n - 1 incremental steps.
        for index in range(max(continuation_ids.shape[1] - 1, 0)):
            next_input = continuation_ids[:, index : index + 1]
            output = model(
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = getattr(output, "past_key_values", None)
            if past_key_values is None:
                raise RuntimeError("model stopped returning past_key_values")


def run_nocache_generation_loop(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
) -> None:
    current_ids = prompt_ids
    with torch.inference_mode():
        for index in range(continuation_ids.shape[1]):
            model(input_ids=current_ids, use_cache=False)
            next_input = continuation_ids[:, index : index + 1]
            current_ids = torch.cat([current_ids, next_input], dim=1)


def run_native_blt_nocache_generation_loop(
    model: torch.nn.Module,
    patcher: object,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
) -> None:
    current_ids = prompt_ids
    with torch.inference_mode():
        for index in range(continuation_ids.shape[1]):
            patch_lengths, _ = patcher.patch(current_ids, include_next_token=True)
            model(current_ids, patch_lengths=patch_lengths)
            next_input = continuation_ids[:, index : index + 1]
            current_ids = torch.cat([current_ids, next_input], dim=1)


def profiler_activities(device: torch.device) -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def measure(
    run_once: Callable[[], None],
    *,
    device: torch.device,
    warmup_runs: int,
) -> Measurement:
    for _ in range(warmup_runs):
        run_once()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    with profile(
        activities=profiler_activities(device),
        with_flops=True,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start

    flops = sum(event.flops or 0 for event in prof.key_averages())
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    return Measurement(
        flops=int(flops),
        elapsed_seconds=elapsed_seconds,
        peak_cuda_memory_bytes=peak_memory,
    )


def is_unrecognized_blt_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "model type `blt`" in message
        and "does not recognize this architecture" in message
    )


def benchmark_transformers_model(
    name: str,
    model_id: str,
    *,
    prompt: str,
    continuation: str,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
    warmup_runs: int,
    trust_remote_code: bool,
    fallback_nocache: bool,
) -> ModelResult:
    model, tokenizer = load_transformers_model_and_tokenizer(
        model_id,
        dtype=dtype,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    prompt_ids = encode_text(tokenizer, prompt, device, add_special_tokens=True)
    continuation_ids = encode_text(
        tokenizer,
        continuation,
        device,
        add_special_tokens=False,
    )
    warnings: list[str] = []

    def run_with_mode(selected_mode: str) -> Measurement:
        if selected_mode == "cached":
            return measure(
                lambda: run_cached_generation_loop(model, prompt_ids, continuation_ids),
                device=device,
                warmup_runs=warmup_runs,
            )
        return measure(
            lambda: run_nocache_generation_loop(model, prompt_ids, continuation_ids),
            device=device,
            warmup_runs=warmup_runs,
        )

    mode_used = mode
    try:
        measurement = run_with_mode(mode)
    except Exception as exc:
        if mode != "cached" or not fallback_nocache:
            raise
        warnings.append(f"cached mode failed with {type(exc).__name__}: {exc}")
        warnings.append("reran this model with nocache mode")
        mode_used = "nocache"
        measurement = run_with_mode(mode_used)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ModelResult(
        name=name,
        model_id=model_id,
        backend="transformers",
        mode_requested=mode,
        mode_used=mode_used,
        dtype=dtype_name(dtype),
        device=str(device),
        prompt_tokens=prompt_ids.shape[1],
        continuation_tokens=continuation_ids.shape[1],
        continuation_bytes=len(continuation.encode("utf-8")),
        flops=measurement.flops,
        elapsed_seconds=measurement.elapsed_seconds,
        peak_cuda_memory_bytes=measurement.peak_cuda_memory_bytes,
        warnings=warnings,
    )


def benchmark_native_blt_model(
    *,
    blt_model_id: str,
    entropy_model_id: str,
    prompt: str,
    continuation: str,
    requested_mode: str,
    dtype: torch.dtype,
    device: torch.device,
    warmup_runs: int,
    fallback_reason: str | None = None,
) -> ModelResult:
    warnings: list[str] = []
    if requested_mode != "nocache":
        warnings.append("native blt backend only exposes no-cache generation in this repo")
    if fallback_reason is not None:
        warnings.append(f"used native blt fallback because {fallback_reason}")

    model, tokenizer, patcher = load_native_blt(
        blt_model_id,
        entropy_model_id,
        dtype=dtype,
        device=device,
    )
    prompt_ids = encode_native_blt_text(tokenizer, prompt, device, add_bos=True)
    continuation_ids = encode_native_blt_text(
        tokenizer,
        continuation,
        device,
        add_bos=False,
    )
    measurement = measure(
        lambda: run_native_blt_nocache_generation_loop(
            model,
            patcher,
            prompt_ids,
            continuation_ids,
        ),
        device=device,
        warmup_runs=warmup_runs,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ModelResult(
        name="blt",
        model_id=blt_model_id,
        backend="native",
        mode_requested=requested_mode,
        mode_used="nocache",
        dtype=dtype_name(dtype),
        device=str(device),
        prompt_tokens=prompt_ids.shape[1],
        continuation_tokens=continuation_ids.shape[1],
        continuation_bytes=len(continuation.encode("utf-8")),
        flops=measurement.flops,
        elapsed_seconds=measurement.elapsed_seconds,
        peak_cuda_memory_bytes=measurement.peak_cuda_memory_bytes,
        warnings=warnings,
    )


def benchmark_blt_model(
    args: argparse.Namespace,
    *,
    prompt: str,
    continuation: str,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
    warmup_runs: int,
    trust_remote_code: bool,
    fallback_nocache: bool,
) -> ModelResult:
    if args.blt_backend == "native":
        return benchmark_native_blt_model(
            blt_model_id=args.native_blt_model,
            entropy_model_id=args.native_blt_entropy_model,
            prompt=prompt,
            continuation=continuation,
            requested_mode=mode,
            dtype=dtype,
            device=device,
            warmup_runs=warmup_runs,
        )

    try:
        return benchmark_transformers_model(
            "blt",
            args.blt_model,
            prompt=prompt,
            continuation=continuation,
            mode=mode,
            dtype=dtype,
            device=device,
            warmup_runs=warmup_runs,
            trust_remote_code=trust_remote_code,
            fallback_nocache=fallback_nocache,
        )
    except Exception as exc:
        if args.blt_backend != "auto" or not is_unrecognized_blt_error(exc):
            raise
        return benchmark_native_blt_model(
            blt_model_id=args.native_blt_model,
            entropy_model_id=args.native_blt_entropy_model,
            prompt=prompt,
            continuation=continuation,
            requested_mode=mode,
            dtype=dtype,
            device=device,
            warmup_runs=warmup_runs,
            fallback_reason=(
                "installed transformers does not recognize model_type blt"
            ),
        )


def format_number(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


def format_flops(value: int) -> str:
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.3f} tflops"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f} gflops"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} mflops"
    return f"{value:,} flops"


def print_summary(results: list[ModelResult]) -> None:
    print("generation flops comparison")
    for result in results:
        print("")
        print(f"{result.name}: {result.model_id}")
        print(f"backend: {result.backend}")
        print(f"mode: {result.mode_used}")
        print(f"prompt tokens: {result.prompt_tokens:,}")
        print(f"continuation tokens: {result.continuation_tokens:,}")
        print(f"continuation bytes: {result.continuation_bytes:,}")
        print(f"flops: {format_flops(result.flops)}")
        print(f"elapsed seconds under profiler: {result.elapsed_seconds:.3f}")
        print(f"peak cuda memory bytes: {format_number(result.peak_cuda_memory_bytes)}")
        for warning in result.warnings:
            print(f"warning: {warning}")

    if len(results) == 2 and results[1].flops:
        ratio = results[0].flops / results[1].flops
        print("")
        print(f"{results[0].name} / {results[1].name} flop ratio: {ratio:.3f}")
    if len({result.mode_used for result in results}) > 1:
        print("")
        print("warning: models used different generation modes, so compare ratios cautiously")
    if any(result.flops == 0 for result in results):
        print("")
        print(
            "warning: torch profiler reported zero flops for at least one model; this can"
            " happen when kernels are not covered by profiler flop accounting"
        )


def main() -> None:
    args = parse_args()
    if RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "this benchmark requires torch and transformers; activate the project"
            " environment or install the dependencies before running it"
        ) from RUNTIME_IMPORT_ERROR
    if args.warmup_runs < 0:
        raise ValueError("warmup-runs must be non-negative")
    if not args.prompt:
        raise ValueError("prompt must be non-empty")
    if not args.continuation:
        raise ValueError("continuation must be non-empty")

    device = torch.device(args.device)
    dtype = choose_dtype(args.dtype, device)
    fallback_nocache = not args.no_fallback_nocache

    blt_result = benchmark_blt_model(
        args,
        prompt=args.prompt,
        continuation=args.continuation,
        mode=args.mode,
        dtype=dtype,
        device=device,
        warmup_runs=args.warmup_runs,
        trust_remote_code=args.trust_remote_code,
        fallback_nocache=fallback_nocache,
    )
    llama_mode = blt_result.mode_used
    llama_result = benchmark_transformers_model(
        "llama",
        args.llama_model,
        prompt=args.prompt,
        continuation=args.continuation,
        mode=llama_mode,
        dtype=dtype,
        device=device,
        warmup_runs=args.warmup_runs,
        trust_remote_code=args.trust_remote_code,
        fallback_nocache=fallback_nocache,
    )
    if llama_mode != args.mode:
        llama_result.warnings.append(
            f"ran llama in {llama_mode} mode to match the blt backend"
        )

    results = [blt_result, llama_result]
    print_summary(results)

    if args.output_json:
        payload = {
            "prompt": args.prompt,
            "continuation": args.continuation,
            "results": [asdict(result) for result in results],
        }
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
