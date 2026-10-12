"""Select the native generation template shipped with a Horizon checkpoint."""

from pathlib import Path


def generation_template(model_path):
    model_path = Path(model_path)
    # 0.9B ships a separate generation template; the pinned 3.7B checkpoint
    # already handles reasoning_content in its main chat template.
    for name in ("chat_template_generation.jinja", "chat_template.jinja"):
        path = model_path / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No native Horizon chat template in {model_path}")
