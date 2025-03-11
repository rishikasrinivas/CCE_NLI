from huggingface_hub import hf_hub_download, login

login(token="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")
file_path = hf_hub_download(
    repo_id="ccenli/BOWMAN",
    filename="bowmanMetrics.zip",
    repo_type="model",  # Use "model" or "space" if applicable
)

print(f"File downloaded to: {file_path}")