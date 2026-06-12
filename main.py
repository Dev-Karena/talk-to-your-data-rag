"""Project entry point for "Talk to Your Data".

Recommended launch:
    streamlit run main.py

Convenience launch:
    python main.py
        -> If not already running under Streamlit, this re-launches itself with
           the Streamlit CLI so users can start the app either way.

Keeping the entry point at the project root ensures ``import app...`` resolves
correctly (the root is on ``sys.path``).
"""

from __future__ import annotations

import sys


def _running_under_streamlit() -> bool:
    """Return ``True`` if executed via ``streamlit run`` (vs. plain ``python``)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001 - any import/runtime issue means "not under streamlit"
        return False


def _relaunch_with_streamlit() -> None:
    """Re-run this file through the Streamlit CLI, preserving extra arguments."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(stcli.main())


def run() -> None:
    """Render the Streamlit UI."""
    # Imported here (not at module top) so the relaunch path doesn't import the
    # full app twice.
    from app.ui.streamlit_app import main

    main()


if __name__ == "__main__":
    if _running_under_streamlit():
        run()
    else:
        # User ran `python main.py` — bootstrap the Streamlit server for them.
        _relaunch_with_streamlit()
