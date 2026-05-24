"""GAZE2CodeToolkit — Streamlit web wrapper.

Run from inside the GAZE2CodeToolkit/ directory:

    streamlit run app.py

The four tabs mirror the five CLI scripts (extract / aoi / visualize /
evaluate_ocr). The "score_expertise" CLI is intentionally not surfaced
as a tab — it is a one-shot batch operation on a CSV the user already
has, so the CLI is the natural entry point.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="GAZE2CodeToolkit",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Tabs are imported AFTER set_page_config so any st.* call they trigger
# at import time happens inside the configured app.
from webapp.tabs import aoi, evaluate, extract, onboard, visualize  # noqa: E402


def main() -> None:
    st.title("GAZE2CodeToolkit")
    st.caption(
        "Eye-tracking → semantic gaze pipeline · Tobii TSV → OCR token AOIs → "
        "fixation×AOI tables. Web wrapper around the CLI scripts in `cli/`."
    )

    with st.sidebar:
        st.subheader("About")
        st.write(
            "Tabs above mirror the four interactive CLI scripts. "
            "Choose a dataset on the **Extract** tab to begin, or jump "
            "straight to **AOI Detection** if you already know which "
            "trial you want."
        )
        st.divider()
        st.write("**Available datasets**")
        try:
            from g2c.parsers import available_datasets
            for d in available_datasets():
                st.write(f"• `{d}`")
        except Exception as e:
            st.warning(f"Parser package failed to import: {e}")
        st.divider()
        st.caption("Streamlit app · GAZE2CodeToolkit")

    tabs = st.tabs([
        "🔧 Onboard Tobii",
        "📥 Extract",
        "🔎 AOI Detection",
        "🎨 Visualize",
        "✅ Evaluate OCR",
    ])
    with tabs[0]:
        onboard.render()
    with tabs[1]:
        extract.render()
    with tabs[2]:
        aoi.render()
    with tabs[3]:
        visualize.render()
    with tabs[4]:
        evaluate.render()

    _render_footer()


def _render_footer() -> None:
    """Site-wide footer with copyright + author info.

    Rendered as raw HTML inside `st.markdown` so it lays out as a
    single small grey line below the active tab content. Streamlit's
    own elements add too much vertical padding for a footer this
    short.
    """
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.85em;
                    padding: 8px 0 16px;">
            © 2026 Wudao (Dylan) Yang
            &lt;<a href="mailto:dylanyang963@gmail.com"
                  style="color: #888;">dylanyang963@gmail.com</a>&gt;
            · GAZE2CodeToolkit · gaze-based programmer-expertise research
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
