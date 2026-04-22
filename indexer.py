"""Build a FAISS index from a folder of images.

Usage:
    python indexer.py --images /path/to/images --out index_data
"""
import argparse
from search import ImageSearchEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Directory containing images (recursive).")
    ap.add_argument("--out", default="index_data", help="Where to save index + path list.")
    ap.add_argument(
        "--model",
        default="google/siglip2-base-patch16-256",
        help="HuggingFace model id. Try 'google/siglip2-so400m-patch14-384' for higher quality.",
    )
    args = ap.parse_args()

    engine = ImageSearchEngine(model_name=args.model)
    n = engine.build_index(args.images)
    engine.save(args.out)
    print(f"Indexed {n} images -> {args.out}")


if __name__ == "__main__":
    main()