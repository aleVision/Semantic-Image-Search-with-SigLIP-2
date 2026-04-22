"""Gradio demo for semantic image search.

Usage:
    python app.py              # after running indexer.py
"""
import argparse
import gradio as gr
from search import ImageSearchEngine


def build_ui(engine: ImageSearchEngine) -> gr.Blocks:
    def search(text_query: str, image_query, k: float):
        k = int(k)
        if image_query is not None:
            results = engine.search_image(image_query, k=k)
        elif text_query and text_query.strip():
            results = engine.search_text(text_query.strip(), k=k)
        else:
            return []
        return [(p, f"score: {s:.3f}") for p, s in results]

    with gr.Blocks(title="Semantic Image Search (SigLIP 2)") as demo:
        gr.Markdown(
            "# 🔍 Semantic Image Search\n"
            "Search your image library by **text description** *or* by **uploading a reference image**. "
            "Powered by SigLIP 2 + FAISS."
        )
        with gr.Row():
            with gr.Column(scale=1):
                text_in = gr.Textbox(
                    label="Text query",
                    placeholder="e.g. a golden retriever on a beach at sunset",
                    lines=2,
                )
                image_in = gr.Image(label="…or upload a reference image", type="pil")
                k_slider = gr.Slider(1, 30, value=9, step=1, label="Top K results")
                btn = gr.Button("Search", variant="primary")
                gr.Markdown(
                    "**Tip:** If both are given, the image query wins. "
                    "Clear the image to switch back to text search."
                )
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Results", columns=3, object_fit="cover", height=620
                )

        btn.click(search, [text_in, image_in, k_slider], gallery)
        text_in.submit(search, [text_in, image_in, k_slider], gallery)

    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index_data", help="Directory with saved index.")
    ap.add_argument("--model", default="google/siglip2-base-patch16-256")
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    args = ap.parse_args()

    engine = ImageSearchEngine(model_name=args.model)
    engine.load(args.index)
    print(f"Loaded index with {len(engine.paths)} images.")

    build_ui(engine).launch(share=args.share)


if __name__ == "__main__":
    main()