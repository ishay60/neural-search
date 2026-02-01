#!/usr/bin/env python3
"""Download and cache embedding models.

This script pre-downloads the embedding and reranking models
to speed up application startup.
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_embedding_model(model_name: str) -> None:
    """Download and cache an embedding model."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Downloading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Test the model
    test_embedding = model.encode(["Test sentence"])
    logger.info(f"  Dimension: {test_embedding.shape[1]}")
    logger.info(f"  Downloaded successfully!")


def download_reranker_model(model_name: str) -> None:
    """Download and cache a reranker model."""
    from sentence_transformers import CrossEncoder

    logger.info(f"Downloading reranker model: {model_name}")
    model = CrossEncoder(model_name)

    # Test the model
    score = model.predict([["Query", "Document"]])
    logger.info(f"  Test score: {score[0]:.4f}")
    logger.info(f"  Downloaded successfully!")


def main():
    """Download all models."""
    parser = argparse.ArgumentParser(description="Download embedding models")
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model to download",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Reranker model to download",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all common models",
    )
    args = parser.parse_args()

    if args.all:
        # Download all common models
        embedding_models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2",
        ]
        reranker_models = [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
        ]

        for model in embedding_models:
            try:
                download_embedding_model(model)
            except Exception as e:
                logger.error(f"Failed to download {model}: {e}")

        for model in reranker_models:
            try:
                download_reranker_model(model)
            except Exception as e:
                logger.error(f"Failed to download {model}: {e}")
    else:
        download_embedding_model(args.embedding_model)
        download_reranker_model(args.reranker_model)

    logger.info("All models downloaded!")


if __name__ == "__main__":
    main()
