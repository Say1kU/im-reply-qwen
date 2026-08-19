param(
    [string]$ModelId = "Qwen/Qwen3-0.6B",
    [string]$Dataset = "data/seed_train.jsonl",
    [string]$OutputDir = "checkpoints/qwen3-06b-im-lora",
    [int]$MaxSteps = -1
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$datasetPath = if ([System.IO.Path]::IsPathRooted($Dataset)) {
    $Dataset
} else {
    Join-Path $projectRoot $Dataset
}
$outputPath = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $projectRoot $OutputDir
}

$env:MODELSCOPE_HOME = Join-Path $projectRoot ".cache/modelscope-home"
$env:MODELSCOPE_CACHE = Join-Path $projectRoot ".cache/modelscope-cache"
$env:HF_HOME = Join-Path $projectRoot ".cache/huggingface"
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = Join-Path $projectRoot ".cache/matplotlib"

$swift = Get-Command swift -ErrorAction Stop
$arguments = @(
    "sft",
    "--model", $ModelId,
    "--tuner_type", "lora",
    "--dataset", $datasetPath,
    "--torch_dtype", "bfloat16",
    "--num_train_epochs", "3",
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--learning_rate", "1e-4",
    "--lora_rank", "8",
    "--lora_alpha", "32",
    "--target_modules", "all-linear",
    "--gradient_accumulation_steps", "8",
    "--logging_steps", "2",
    "--save_steps", "20",
    "--save_total_limit", "2",
    "--max_length", "1024",
    "--warmup_ratio", "0.05",
    "--dataloader_num_workers", "0",
    "--output_dir", $outputPath
)

if ($MaxSteps -gt 0) {
    $arguments += @("--max_steps", $MaxSteps.ToString())
}

& $swift.Source @arguments
exit $LASTEXITCODE

