#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/prepare_models.sh HF_REPO PINNED_COMMIT OUTPUT_DIR LLAMA_CPP_REPO

Downloads the requested Hugging Face revision only when explicitly invoked, then
converts the same local source snapshot to MLX float16 and GGUF F16. Requires
`hf`, `mlx_lm`, `python`, and a cloned llama.cpp checkout with conversion deps.
USAGE
}
if [[ $# -ne 4 ]]; then usage >&2; exit 2; fi
repo="$1" revision="$2" output_dir="$3" llama_repo="$4"
if [[ ! "$revision" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "Revision must be a full 40-character commit SHA." >&2; exit 2
fi
if ! command -v hf >/dev/null || ! command -v mlx_lm >/dev/null || ! command -v python3 >/dev/null; then
  echo "Install huggingface_hub, mlx-lm, and Python first (mlx_lm requires Apple Silicon)." >&2; exit 2
fi
converter="$llama_repo/convert_hf_to_gguf.py"
if [[ ! -f "$converter" ]]; then echo "llama.cpp converter not found: $converter" >&2; exit 2; fi
mkdir -p "$output_dir"
source_dir="$output_dir/source"
mlx_dir="$output_dir/mlx"
gguf_file="$output_dir/model.f16.gguf"
hf download "$repo" --revision "$revision" --local-dir "$source_dir"
mlx_lm.convert --hf-path "$source_dir" --mlx-path "$mlx_dir" --dtype float16
python3 "$converter" "$source_dir" --outfile "$gguf_file" --outtype f16
converter_revision="$(git -C "$llama_repo" rev-parse HEAD 2>/dev/null || echo unavailable)"
python3 - "$repo" "$revision" "$source_dir" "$mlx_dir" "$gguf_file" "$output_dir/model-manifest.json" "$converter_revision" <<'PY'
import hashlib, json, pathlib, subprocess, sys
repo, revision, source, mlx, gguf, manifest, llama_revision = sys.argv[1:]
def version(package):
    try:
        from importlib.metadata import version as get_version
        return get_version(package)
    except Exception:
        return "unavailable"
def sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
source_path=pathlib.Path(source)
config={}
tokenizer_config={}
for filename, target in (("config.json",config),("tokenizer_config.json",tokenizer_config)):
    candidate=source_path/filename
    if candidate.is_file(): target.update(json.loads(candidate.read_text(encoding="utf-8")))
weights=[p for p in source_path.rglob("*.safetensors") if p.is_file()] or [p for p in source_path.rglob("*.bin") if p.is_file()]
manifest_data={
    "schema_version":"1.0", "source_model_id":repo, "source_revision":revision,
    "source_path":str(pathlib.Path(source).resolve()), "mlx_path":str(pathlib.Path(mlx).resolve()),
    "gguf_path":str(pathlib.Path(gguf).resolve()), "source_dtype":config.get("torch_dtype", "not declared in config.json"),
    "mlx_dtype":"float16", "gguf_dtype":"f16", "quantization":None,
    "conversion_commands":["mlx_lm.convert --hf-path SOURCE --mlx-path MLX --dtype float16", "python convert_hf_to_gguf.py SOURCE --outfile MODEL.f16.gguf --outtype f16"],
    "conversion_versions":{"mlx-lm":version("mlx-lm"),"transformers":version("transformers"),"llama_cpp_revision":llama_revision},
    "architecture":{"model_type":config.get("model_type"),"architectures":config.get("architectures")},
    "tokenizer_metadata":{"bos_token":tokenizer_config.get("bos_token"),"eos_token":tokenizer_config.get("eos_token"),"bos_token_id":tokenizer_config.get("bos_token_id"),"eos_token_id":tokenizer_config.get("eos_token_id"),"chat_template":tokenizer_config.get("chat_template")},
    "source_weight_sha256":{str(p.relative_to(source_path)):sha256(p) for p in weights},
    "mlx_weight_sha256":{str(p.relative_to(pathlib.Path(mlx))):sha256(p) for p in pathlib.Path(mlx).rglob("*.safetensors") if p.is_file()},
    "gguf_sha256":sha256(pathlib.Path(gguf)),
}
pathlib.Path(manifest).write_text(json.dumps(manifest_data,indent=2)+"\n",encoding="utf-8")
PY
echo "Prepared artifacts and provenance manifest at $output_dir"
