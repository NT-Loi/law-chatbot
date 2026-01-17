import re
import hashlib


def slugify_model_name(model_name: str) -> str:
    """Converts model name to a slug suitable for collection names."""
    slug = model_name.split("/")[-1]
    slug = re.sub(r'[^a-zA-Z0-9]', '_', slug).lower()
    return slug.strip("_")


def get_collection_name(source: str, model_name: str) -> str:
    """Generates a collection name based on source and model."""
    return f"{source}_{slugify_model_name(model_name)}"


def get_point_id(doc_id: str, hierarchy_path: str) -> str:
    """
    Generate unique point ID (MD5 Hash) based on document ID and hierarchy path.
    Used for VBQPPL and Pháp Điển collections.
    """
    safe_doc_id = str(doc_id) if doc_id else "unknown_doc"
    safe_path = str(hierarchy_path) if hierarchy_path else "general"
    raw_combination = f"{safe_doc_id}_{safe_path}"
    return hashlib.md5(raw_combination.encode('utf-8')).hexdigest()


def get_alqac_point_id(doc_id: str, article_id: str) -> str:
    """
    Generate unique point ID (MD5 Hash) for ALQAC-2025 collection.
    Uses document ID and article ID as the combination key.
    """
    safe_doc_id = str(doc_id) if doc_id else "unknown_doc"
    safe_article_id = str(article_id) if article_id else "unknown_article"
    raw_combination = f"{safe_doc_id}_{safe_article_id}"
    return hashlib.md5(raw_combination.encode('utf-8')).hexdigest()