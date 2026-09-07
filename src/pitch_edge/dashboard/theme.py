"""Design tokens + Plotly template for the dashboard — dark mode, validated palette.

The categorical slots are the reference palette's *dark-surface* steps (validated for adjacent-pair
colour-vision separation on `#1a1a19`), not a brightness flip of the light set. Outcomes map
home → slot 1 (blue), away → slot 2 (orange), draw → neutral ink, so the two teams are never in the
same hue family. Sequential = single-hue blue; diverging = blue ↔ red with a neutral dark midpoint;
status colours are reserved for status and never reused for a series.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
OUTCOME = {"home": "#3987e5", "away": "#d95926", "draw": "#898781"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQ_BLUE = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
DIVERGING = [[0.0, "#3987e5"], [0.5, "#383835"], [1.0, "#e66767"]]
INK = {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"}
SURFACE = "#1a1a19"
PAGE = "#0d0d0d"


def install_template() -> None:
    layout = go.Layout(
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "size": 13, "color": INK["primary"]},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        margin={"l": 48, "r": 24, "t": 56, "b": 44},
        title={"font": {"size": 15, "color": INK["primary"]}, "x": 0, "xanchor": "left"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": INK["secondary"]},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="x unified",
        hoverlabel={"bgcolor": "#262624", "bordercolor": INK["axis"], "font": {"color": INK["primary"], "size": 12}},
        xaxis={
            "gridcolor": INK["grid"],
            "zerolinecolor": INK["axis"],
            "linecolor": INK["axis"],
            "tickfont": {"color": INK["muted"]},
            "title": {"font": {"color": INK["secondary"]}},
            "showline": False,
        },
        yaxis={
            "gridcolor": INK["grid"],
            "zerolinecolor": INK["axis"],
            "linecolor": INK["axis"],
            "tickfont": {"color": INK["muted"]},
            "title": {"font": {"color": INK["secondary"]}},
            "showline": False,
        },
        colorscale={"sequential": [[i / 6, c] for i, c in enumerate(SEQ_BLUE)], "diverging": DIVERGING},
        geo={
            "bgcolor": SURFACE,
            "landcolor": "#262624",
            "lakecolor": SURFACE,
            "oceancolor": SURFACE,
            "showocean": True,
            "coastlinecolor": INK["axis"],
            "countrycolor": INK["axis"],
            "showcountries": True,
        },
        bargap=0.28,
    )
    pio.templates["pitch_edge"] = go.layout.Template(
        layout=layout,
        data={
            "bar": [go.Bar(marker={"line": {"width": 0}})],
            "scatter": [go.Scatter(line={"width": 2}, marker={"size": 8})],
        },
    )
    pio.templates.default = "pitch_edge"


CSS = f"""
<style>
  .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }}
  h1 {{ font-size: 1.9rem; letter-spacing: -0.01em; margin-bottom: 0.2rem; }}
  h2, h3 {{ letter-spacing: -0.005em; }}
  [data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid rgba(255,255,255,0.10); border-radius: 10px; padding: 12px 16px; }}
  [data-testid="stMetricLabel"] {{ color: {INK["secondary"]}; font-size: 0.8rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.55rem; color: {INK["primary"]}; }}
  .pe-note {{ color: {INK["secondary"]}; font-size: 0.9rem; border-left: 3px solid {SERIES[0]}; padding: 6px 12px; background: {SURFACE}; border-radius: 6px; margin: 4px 0 12px 0; }}
  .pe-warn {{ color: {INK["primary"]}; font-size: 0.9rem; border-left: 3px solid {STATUS["warning"]}; padding: 6px 12px; background: #2a2416; border-radius: 6px; margin: 4px 0 12px 0; }}
  .pe-pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; background: #17304f; color: #86b6ef; margin-right: 6px; }}
  div[data-baseweb="tab-list"] {{ gap: 4px; }}
  button[data-baseweb="tab"] {{ font-size: 0.92rem; }}
  [data-testid="stSidebar"] {{ background: {SURFACE}; }}
</style>
"""
