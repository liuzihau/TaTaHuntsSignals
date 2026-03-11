#!/usr/bin/env python3
"""
Encode all articles with FinBERT embeddings (768-dim).

Usage:
    python3 run_finbert.py [--batch-size 32] [--device cuda]

Output:
    - articles table: embedding column populated with 768-dim vectors
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

from embedding.finbert_encoder import FinBertEncoder
from storage.postgres_client import PostgreSQLClient


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Encode articles with FinBERT")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu (default: cuda, falls back to cpu if unavailable)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit articles to encode (for testing)",
    )
    parser.add_argument(
        "--force-reencode",
        action="store_true",
        help="Re-encode articles that already have embeddings",
    )
    return parser.parse_args()


def main() -> None:
    """Main FinBERT encoding pipeline."""
    args = parse_args()
    
    LOGGER.info(f"🚀 TaTaHunts FinBERT Encoding v1")
    LOGGER.info(f"   Batch size: {args.batch_size}")
    LOGGER.info(f"   Device: {args.device}")
    LOGGER.info(f"   Force reencode: {args.force_reencode}")
    
    # Initialize clients
    pg = PostgreSQLClient()
    encoder = FinBertEncoder(device=args.device, batch_size=args.batch_size)
    
    try:
        # Fetch articles needing embedding
        LOGGER.info("Fetching articles needing embedding...")
        articles = pg.get_articles_needing_embedding(limit=args.limit or 10000)
        
        if not articles:
            LOGGER.warning("⚠️ No articles found needing embedding!")
            LOGGER.info("   Try running: python3 crawl_recent_news.py")
            return
        
        LOGGER.info(f"Found {len(articles)} articles to encode")
        
        # Prepare texts for encoding
        article_ids = [a["id"] for a in articles]
        texts = [
            f"{a.get('title', '')} {a.get('body', '')}"[:2000]  # Truncate to 2000 chars
            for a in articles
        ]
        
        # Encode batch
        LOGGER.info(f"Encoding {len(texts)} articles with FinBERT (768-dim)...")
        embeddings = encoder.encode_batch(texts, batch_size=args.batch_size)
        
        if len(embeddings) != len(article_ids):
            LOGGER.error(f"Encoding mismatch: {len(embeddings)} embeddings vs {len(article_ids)} IDs")
            sys.exit(1)
        
        # Store embeddings to PostgreSQL
        LOGGER.info("Storing embeddings to PostgreSQL...")
        success = pg.store_embeddings(
            article_ids=article_ids,
            embeddings=embeddings,
            embedding_model="finbert",
            embedding_dimension=768,
        )
        
        if not success:
            LOGGER.error("Failed to store embeddings")
            sys.exit(1)
        
        # Verify
        with pg._require_conn().cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM articles WHERE has_embedding = TRUE")
            embedded_count = cur.fetchone()[0]
        
        LOGGER.info("=" * 70)
        LOGGER.info("✅ ENCODING COMPLETE")
        LOGGER.info("=" * 70)
        LOGGER.info(f"Total embedded articles: {embedded_count}")
        LOGGER.info(f"Next: Run feature aggregation (Phase 6)")
        
    except Exception as e:
        LOGGER.error(f"❌ Encoding failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        pg.close()


if __name__ == "__main__":
    main()
