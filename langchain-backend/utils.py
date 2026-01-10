import re

def slugify_model_name(model_name: str) -> str:
    """Converts model name to a slug suitable for collection names."""
    slug = model_name.split("/")[-1] # Take the name after the org
    slug = re.sub(r'[^a-zA-Z0-9]', '_', slug).lower()
    return slug.strip("_")

def get_collection_name(source: str, model_name: str) -> str:
    """Generates a collection name based on source and model."""
    return f"{source}_{slugify_model_name(model_name)}"