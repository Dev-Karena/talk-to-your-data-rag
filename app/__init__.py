"""Talk to Your Data — local Retrieval-Augmented Generation application.

Top-level package. Layers:
    * ``app.config``   — typed settings loaded from ``.env``.
    * ``app.utils``    — logging and upload validation.
    * ``app.rag``      — ingestion pipeline (load, clean, chunk, embed, store).
    * ``app.services`` — query-time orchestration (retrieve, build context, LLM).
    * ``app.ui``       — Streamlit presentation layer.
"""

__version__ = "1.0.0"
