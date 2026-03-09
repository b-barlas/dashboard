"""Global Streamlit CSS styles."""

from __future__ import annotations

from ui.theme import (
    ACCENT,
    NEGATIVE,
    NEON_BLUE,
    NEON_PURPLE,
    POSITIVE,
    PRIMARY_BG,
    TEXT_LIGHT,
    TEXT_MUTED,
    WARNING,
)


def app_css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

/* Global styles */
.stApp {{
    background-color: {PRIMARY_BG};
    color: {TEXT_LIGHT};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
}}

/* Global typography system (all tabs) */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
    font-family: 'Space Grotesk', 'Manrope', sans-serif !important;
    letter-spacing: 0.1px;
}}
.stMarkdown p,
.stMarkdown li,
.stMarkdown label,
.stCaption,
div[data-testid="stCaptionContainer"] {{
    font-family: 'Manrope', 'Segoe UI', sans-serif !important;
}}
.stMarkdown summary {{
    font-family: 'Space Grotesk', 'Manrope', sans-serif !important;
}}

/* Market shared typography/components */
.market-intro-title {{
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
    letter-spacing: 0.2px;
}}
.market-intro-body {{
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.9rem;
    line-height: 1.62;
    margin-top: 6px;
}}
.market-section-title {{
    color: inherit;
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 1.62rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    margin-bottom: 0.45rem;
}}
.market-gauge-chip-wrap {{
    text-align: center;
    margin-top: -6px;
}}
.market-gauge-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 2px 10px;
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.15px;
}}
.market-inline-chip {{
    display: inline-block;
    border-radius: 999px;
    padding: 3px 10px;
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.15px;
}}
.market-note-box {{
    border-radius: 10px;
    padding: 8px 10px;
    margin: 0 0 0.6rem 0;
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.83rem;
    line-height: 1.55;
}}
.market-header-card {{
    background:
        linear-gradient(180deg, rgba(0, 212, 255, 0.06), transparent 18%),
        linear-gradient(160deg, rgba(3, 8, 15, 0.98), rgba(1, 4, 9, 0.99));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: 14px 18px 10px;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 18px 40px rgba(0,0,0,0.26);
    overflow: hidden;
    position: relative;
    margin-bottom: 16px;
}}
.market-header-card::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    opacity: 0.9;
}}
.market-header-card::after {{
    content: "";
    position: absolute;
    top: 18px;
    right: -28px;
    width: 140px;
    height: 140px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0, 212, 255, 0.05), transparent 68%);
    opacity: 0.18;
    pointer-events: none;
}}
.market-header-card--muted {{
    --header-accent: rgba(148, 163, 184, 0.9);
}}
.market-header-card--sentiment {{
    background:
        linear-gradient(180deg, rgba(255, 209, 102, 0.06), transparent 18%),
        linear-gradient(160deg, rgba(8, 9, 14, 0.98), rgba(5, 5, 8, 0.99));
}}
.market-header-title,
.market-header-main,
.market-header-band,
.market-header-band-guides,
.market-header-note,
.market-header-sentiment-scale {{
    position: relative;
    z-index: 1;
}}
.market-header-title {{
    color: #D7E4F2;
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.72px;
    text-transform: uppercase;
}}
.market-header-info {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 0.92rem;
    height: 0.92rem;
    margin-left: 0.28rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.16);
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    line-height: 1;
    vertical-align: middle;
}}
.market-header-main {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 10px;
}}
.market-header-main--sentiment {{
    align-items: center;
}}
.market-header-value {{
    color: #F8FAFC;
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.6px;
}}
.market-header-pill {{
    display: inline-flex;
    align-items: center;
    align-self: center;
    border-radius: 999px;
    padding: 5px 11px;
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    flex-shrink: 0;
}}
.market-header-pill--muted {{
    color: {TEXT_MUTED};
    border-color: rgba(255,255,255,0.08);
}}
.market-header-band,
.market-header-sentiment-scale {{
    position: relative;
    margin-top: 10px;
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
}}
.market-header-band-seg,
.market-header-sentiment-seg {{
    position: absolute;
    top: 0;
    bottom: 0;
}}
.market-header-band-seg--neg {{
    left: 0;
    width: 44%;
    background: linear-gradient(90deg, rgba(255, 51, 102, 0.98), rgba(255, 85, 128, 0.88));
}}
.market-header-band-seg--neutral {{
    left: 44%;
    width: 12%;
    background: rgba(255, 209, 102, 0.88);
}}
.market-header-band-seg--pos {{
    right: 0;
    width: 44%;
    background: linear-gradient(90deg, rgba(36, 238, 141, 0.88), rgba(0, 255, 136, 0.98));
}}
.market-header-sentiment-seg--fear {{
    left: 0;
    width: 20%;
    background: rgba(255, 51, 102, 0.98);
}}
.market-header-sentiment-seg--caution {{
    left: 20%;
    width: 20%;
    background: rgba(255, 102, 102, 0.9);
}}
.market-header-sentiment-seg--neutral {{
    left: 40%;
    width: 20%;
    background: rgba(255, 209, 102, 0.88);
}}
.market-header-sentiment-seg--greed {{
    left: 60%;
    width: 20%;
    background: rgba(36, 238, 141, 0.86);
}}
.market-header-sentiment-seg--extreme {{
    left: 80%;
    width: 20%;
    background: rgba(0, 255, 136, 0.96);
}}
.market-header-band-marker,
.market-header-sentiment-marker {{
    position: absolute;
    top: 50%;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    border: 2px solid rgba(3, 8, 15, 0.95);
}}
.market-header-band-guides {{
    display: flex;
    justify-content: space-between;
    gap: 8px;
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.12px;
    text-transform: uppercase;
}}
.market-header-note {{
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.7rem;
    line-height: 1.3;
    flex: 0 0 auto;
}}
.market-orbital-card {{
    background:
        radial-gradient(circle at top center, rgba(0, 212, 255, 0.07), transparent 36%),
        linear-gradient(160deg, rgba(4, 10, 18, 0.96), rgba(1, 4, 9, 0.99));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 16px 16px 12px;
    min-height: 314px;
    height: 314px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 16px 36px rgba(0,0,0,0.28);
    overflow: hidden;
    position: relative;
}}
.market-orbital-title,
.market-orbital-stage,
.market-orbital-guides,
.market-orbital-note,
.market-top-footer {{
    position: relative;
    z-index: 1;
}}
.market-orbital-title-row {{
    position: relative;
    display: flex;
    align-items: flex-start;
    justify-content: flex-start;
    min-height: 28px;
    flex-shrink: 0;
}}
.market-orbital-title {{
    color: #E5E7EB;
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.74px;
    text-transform: uppercase;
    flex-shrink: 0;
}}
.market-orbital-topmeta {{
    position: absolute;
    top: 0;
    right: 0;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 3px 10px;
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.12px;
    background: rgba(255,255,255,0.04);
    white-space: nowrap;
    flex-shrink: 0;
}}
.market-orbital-stage {{
    position: relative;
    height: 184px;
    flex-shrink: 0;
}}
.market-orbital-svg {{
    width: 100%;
    height: 100%;
    display: block;
}}
.market-orbital-center {{
    position: absolute;
    inset: 88px 0 auto 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    pointer-events: none;
}}
.market-orbital-value-row {{
    display: flex;
    align-items: baseline;
    gap: 6px;
    justify-content: center;
    min-width: 0;
    width: 100%;
    text-align: center;
}}
.market-orbital-value {{
    color: #F8FAFC;
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 2.36rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.6px;
}}
.market-orbital-guides {{
    display: flex;
    justify-content: space-between;
    gap: 8px;
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.1px;
    flex-shrink: 0;
    margin-top: -10px;
}}
.market-orbital-note {{
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.67rem;
    line-height: 1.28;
    flex: 0 0 auto;
}}
.market-top-unit {{
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
}}
.market-top-meta {{
    display: flex;
    justify-content: space-between;
    gap: 10px;
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.72rem;
}}
.market-top-meta strong {{
    color: #F8FAFC;
    font-weight: 700;
}}
.market-top-footer {{
    margin-top: auto;
    flex-shrink: 0;
}}
.market-statline {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0;
    border-top: 1px solid rgba(255,255,255,0.06);
    padding-top: 4px;
}}
.market-statline-item {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1px;
    min-width: 0;
    padding: 0 6px;
    text-align: center;
}}
.market-statline-item + .market-statline-item {{
    border-left: 1px solid rgba(255,255,255,0.06);
}}
.market-statline-label {{
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.46rem;
    font-weight: 700;
    letter-spacing: 0.28px;
    text-transform: uppercase;
    line-height: 1;
}}
.market-statline-value {{
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    line-height: 1;
}}
.market-details summary {{
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 0.92rem;
    font-weight: 600;
    cursor: pointer;
}}
.market-details-body {{
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.84rem;
    line-height: 1.72;
    padding: 0.4rem 0.2rem;
}}
.market-criteria-chip {{
    padding: 2px 10px;
    border-radius: 999px;
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15px;
}}

/* Shared page primitives */
.app-page-header {{
    margin-bottom: 1rem;
}}
.app-page-title {{
    color: {ACCENT};
    margin: 0 0 0.35rem 0;
}}
.app-page-title--hero {{
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, {ACCENT}, {NEON_BLUE}, {NEON_PURPLE});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}}
.app-page-subtitle {{
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 1rem;
    line-height: 1.6;
    margin: 0 0 0.8rem 0;
}}
.app-intro-card {{
    background: linear-gradient(140deg, rgba(0, 0, 0, 0.76), rgba(8, 18, 30, 0.9));
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}}
.app-intro-title {{
    color: {ACCENT};
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.2px;
}}
.app-intro-body {{
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.9rem;
    line-height: 1.65;
    margin-top: 6px;
}}
.app-help-details {{
    margin: 0.15rem 0 0.75rem 0;
}}
.app-help-details summary {{
    color: {ACCENT};
    cursor: pointer;
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 0.94rem;
    font-weight: 600;
}}
.app-help-body {{
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.85rem;
    line-height: 1.7;
    margin-top: 0.45rem;
}}
.app-sidebar-title-wrap {{
    text-align: center;
    margin: 8px 0 10px 0;
}}
.app-sidebar-title {{
    color: {ACCENT};
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 1.08rem;
    font-weight: 700;
    letter-spacing: 0.2px;
}}
.app-sidebar-panel {{
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 10px;
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    line-height: 1.58;
    background: linear-gradient(140deg, rgba(6, 10, 18, 0.94), rgba(4, 8, 14, 0.94));
}}
.app-sidebar-panel--accent {{
    border: 1px solid rgba(0, 212, 255, 0.24);
}}
.app-sidebar-panel--neutral {{
    border: 1px solid rgba(255, 255, 255, 0.12);
}}
.app-sidebar-panel--positive {{
    border: 1px solid rgba(0, 255, 136, 0.2);
}}
.app-sidebar-panel--warning {{
    border: 1px solid rgba(255, 209, 102, 0.22);
}}
.app-sidebar-panel-title {{
    color: {NEON_BLUE};
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 0.84rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    margin-bottom: 4px;
}}
.app-sidebar-panel-body {{
    color: {TEXT_MUTED};
    font-size: 0.78rem;
}}
.app-status-pill {{
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 2.25rem;
    border-radius: 999px;
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    margin-top: 0.2rem;
}}
.app-status-pill--positive {{
    color: {POSITIVE};
}}
.app-status-pill--warning {{
    color: {WARNING};
}}
.app-status-pill--neutral {{
    color: {TEXT_MUTED};
}}
.app-kpi-grid {{
    display: grid;
    grid-template-columns: repeat(var(--app-kpi-columns, 4), minmax(0, 1fr));
    gap: 10px;
    margin: 8px 0 12px 0;
}}
.app-kpi-grid--center-last {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}}
.app-kpi-grid--center-last > .app-kpi-card {{
    width: calc((100% - (var(--app-kpi-columns, 4) - 1) * 10px) / var(--app-kpi-columns, 4));
    max-width: calc((100% - (var(--app-kpi-columns, 4) - 1) * 10px) / var(--app-kpi-columns, 4));
    flex: 0 0 calc((100% - (var(--app-kpi-columns, 4) - 1) * 10px) / var(--app-kpi-columns, 4));
}}
.app-kpi-card {{
    border: 1px solid rgba(0, 212, 255, 0.16);
    border-radius: 12px;
    padding: 12px 14px;
    background: linear-gradient(140deg, rgba(0, 0, 0, 0.72), rgba(10, 18, 30, 0.88));
    min-height: var(--app-kpi-card-min-height, auto);
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: var(--app-kpi-align, left);
}}
.app-kpi-label {{
    color: {TEXT_MUTED};
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}}
.app-kpi-value {{
    color: {ACCENT};
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 4px;
}}
.app-kpi-sub {{
    color: {TEXT_MUTED};
    font-size: 0.8rem;
    line-height: 1.45;
    margin-top: 4px;
}}
.app-badge-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 2px 0 10px 0;
}}
.app-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 3px 10px;
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.15px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    background: rgba(255, 255, 255, 0.03);
    color: {TEXT_MUTED};
}}
.app-chip-dot {{
    line-height: 1;
}}
.app-chip--accent {{
    color: {ACCENT};
    border-color: rgba(0, 212, 255, 0.28);
    background: rgba(0, 212, 255, 0.06);
}}
.app-chip--positive {{
    color: {POSITIVE};
    border-color: rgba(0, 255, 136, 0.35);
    background: rgba(0, 255, 136, 0.08);
}}
.app-chip--warning {{
    color: {WARNING};
    border-color: rgba(255, 209, 102, 0.35);
    background: rgba(255, 209, 102, 0.08);
}}
.app-chip--negative {{
    color: {NEGATIVE};
    border-color: rgba(255, 51, 102, 0.35);
    background: rgba(255, 51, 102, 0.08);
}}
.app-chip--neutral {{
    color: {TEXT_MUTED};
    border-color: rgba(255, 255, 255, 0.15);
    background: rgba(255, 255, 255, 0.03);
}}
.app-insight-card {{
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-left: 4px solid {ACCENT};
    border-radius: 12px;
    padding: 14px 16px;
    background: linear-gradient(140deg, rgba(0, 0, 0, 0.76), rgba(8, 18, 32, 0.92));
    margin: 10px 0 14px 0;
}}
.app-insight-card--accent {{
    border-left-color: {ACCENT};
}}
.app-insight-card--positive {{
    border-left-color: {POSITIVE};
}}
.app-insight-card--warning {{
    border-left-color: {WARNING};
}}
.app-insight-card--negative {{
    border-left-color: {NEGATIVE};
}}
.app-insight-card--neutral {{
    border-left-color: rgba(255, 255, 255, 0.24);
}}
.app-insight-title {{
    color: {ACCENT};
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 6px;
}}
.app-insight-body {{
    color: {TEXT_MUTED};
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    font-size: 0.87rem;
    line-height: 1.6;
}}
.app-insight-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}}

/* Global heading scale */
h1, .stMarkdown h1 {{
    font-size: 2.35rem !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
}}
h2, .stMarkdown h2 {{
    font-size: 1.62rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
}}
h3, .stMarkdown h3 {{
    font-size: 1.22rem !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
}}
h4, .stMarkdown h4 {{
    font-size: 1.02rem !important;
    font-weight: 650 !important;
    line-height: 1.3 !important;
}}
h5, .stMarkdown h5, h6, .stMarkdown h6 {{
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    line-height: 1.35 !important;
}}

/* Custom scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {PRIMARY_BG}; }}
::-webkit-scrollbar-thumb {{ background: linear-gradient(180deg, {NEON_BLUE}, {NEON_PURPLE}); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: linear-gradient(180deg, {POSITIVE}, {NEON_BLUE}); }}

/* Metric delta colors */
.metric-delta-positive {{
    color: {POSITIVE};
    font-weight: 600;
    font-size: 0.85rem;
    text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
}}
.metric-delta-negative {{
    color: {NEGATIVE};
    font-weight: 600;
    font-size: 0.85rem;
    text-shadow: 0 0 10px rgba(255, 51, 102, 0.5);
}}

/* Glassmorphism titles */
h1.title {{
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, {ACCENT}, {NEON_BLUE}, {NEON_PURPLE});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
    letter-spacing: -0.5px;
}}

p.subtitle {{
    font-size: 1.05rem;
    color: {TEXT_MUTED};
    margin-top: 0;
    margin-bottom: 2rem;
}}

/* Glassmorphism metric cards */
.metric-card {{
    background: rgba(0, 0, 0, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}}
.metric-card:hover {{
    border-color: rgba(0, 212, 255, 0.4);
    box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    transform: translateY(-2px);
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 200%;
    height: 1px;
    background: linear-gradient(90deg, transparent, {NEON_BLUE}, transparent);
    animation: shimmer 3s ease infinite;
}}

@keyframes shimmer {{
    0% {{ left: -100%; }}
    100% {{ left: 100%; }}
}}

.metric-label {{
    font-size: 0.8rem;
    color: {TEXT_MUTED};
    margin-bottom: 8px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 500;
}}
.metric-value {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {ACCENT};
    text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
}}

/* Glassmorphism panel boxes */
.panel-box {{
    background: rgba(0, 0, 0, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 32px;
    border: 1px solid rgba(0, 212, 255, 0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}}
.panel-box::after {{
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
}}

/* Neon glow signal cards */
.signal-long {{
    background: rgba(0, 255, 136, 0.08);
    border: 1px solid rgba(0, 255, 136, 0.3);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
}}
.signal-short {{
    background: rgba(255, 51, 102, 0.08);
    border: 1px solid rgba(255, 51, 102, 0.3);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 0 20px rgba(255, 51, 102, 0.1);
}}
.signal-wait {{
    background: rgba(255, 209, 102, 0.08);
    border: 1px solid rgba(255, 209, 102, 0.3);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 0 20px rgba(255, 209, 102, 0.1);
}}

/* Enhanced table styling */
.table-container {{ overflow-x: auto; }}
table.dataframe {{
    width: 100% !important;
    border-collapse: collapse;
    background-color: rgba(0, 0, 0, 0.9);
    backdrop-filter: blur(10px);
}}
table.dataframe thead tr {{
    background: linear-gradient(90deg, rgba(0, 212, 255, 0.1), rgba(178, 75, 243, 0.1));
}}
table.dataframe th {{
    color: {NEON_BLUE};
    padding: 12px;
    text-align: left;
    font-size: 0.85rem;
    border-bottom: 1px solid rgba(0, 212, 255, 0.2);
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-weight: 600;
}}
table.dataframe td {{
    padding: 10px 12px;
    font-size: 0.9rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    color: {TEXT_LIGHT};
    transition: background 0.2s;
}}
table.dataframe tbody tr:hover {{
    background-color: rgba(0, 212, 255, 0.05);
}}

/* Streamlit tab styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    background: rgba(0, 0, 0, 0.8);
    border-radius: 10px;
    padding: 3px;
    flex-wrap: nowrap;
    overflow-x: auto;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    color: {TEXT_MUTED};
    font-weight: 500;
    padding: 4px 8px;
    font-size: 0.75rem;
    transition: all 0.2s ease;
    white-space: nowrap;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(178, 75, 243, 0.15)) !important;
    color: {ACCENT} !important;
    border-bottom-color: {NEON_BLUE} !important;
}}

/* Streamlit button styling */
.stButton > button {{
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(178, 75, 243, 0.2));
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: {ACCENT};
    border-radius: 14px;
    min-height: 2.95rem;
    padding: 0.7rem 1.05rem;
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.2px;
    transition: all 0.3s ease;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.35), rgba(178, 75, 243, 0.35));
    border-color: {NEON_BLUE};
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
    transform: translateY(-1px);
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {NEON_BLUE}, {NEON_PURPLE});
    border: none;
    color: white;
    font-weight: 700;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.4);
}}

/* Sidebar styling */
section[data-testid="stSidebar"] {{
    background-color: {PRIMARY_BG};
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background-color: rgba(0, 0, 0, 0.9) !important;
    border-color: rgba(0, 212, 255, 0.2) !important;
    color: {TEXT_LIGHT} !important;
    border-radius: 12px !important;
    font-family: 'Manrope', 'Segoe UI', sans-serif !important;
}}

/* Fix dark blue backgrounds on Streamlit widgets */
.stSlider > div,
.stCheckbox > label,
.stRadio > div,
.stNumberInput > div > div > input {{
    background-color: transparent !important;
}}
[data-testid="stMetric"],
[data-testid="stMetricValue"],
[data-testid="column"] {{
    background-color: transparent !important;
}}
div[data-testid="stVerticalBlock"] > div {{
    background-color: transparent !important;
}}
.stSelectbox label,
.stTextInput label,
.stMultiSelect label,
.stSlider label,
.stCheckbox label,
.stNumberInput label {{
    color: {TEXT_MUTED} !important;
}}

/* Expander styling */
.streamlit-expanderHeader {{
    background: rgba(0, 0, 0, 0.7);
    border-radius: 8px;
}}

/* Live ticker bar */
.ticker-bar {{
    background: linear-gradient(90deg, rgba(0, 212, 255, 0.05), rgba(178, 75, 243, 0.05), rgba(0, 212, 255, 0.05));
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 10px;
    padding: 8px 16px;
    margin-bottom: 16px;
    overflow: hidden;
    position: relative;
}}
.ticker-content {{
    display: flex;
    animation: tickerScroll 30s linear infinite;
    white-space: nowrap;
    gap: 40px;
}}
@keyframes tickerScroll {{
    0% {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
}}
.ticker-item {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    color: {TEXT_LIGHT};
}}

/* Heatmap cell */
.heatmap-cell {{
    border-radius: 6px;
    padding: 8px;
    text-align: center;
    transition: transform 0.2s;
    cursor: default;
}}
.heatmap-cell:hover {{
    transform: scale(1.05);
    z-index: 10;
}}

/* Section header */
.god-header {{
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.1), transparent);
    border-left: 3px solid {NEON_BLUE};
    padding: 8px 16px;
    margin: 16px 0;
    border-radius: 0 8px 8px 0;
}}

/* Elite shared components */
.elite-card {{
    background: linear-gradient(140deg, rgba(4, 10, 18, 0.96), rgba(2, 5, 11, 0.96));
    border: 1px solid rgba(0, 212, 255, 0.16);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}}
.elite-label {{
    color: {TEXT_MUTED};
    font-size: 0.72rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    font-weight: 600;
}}
.elite-value {{
    color: {ACCENT};
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: 4px;
}}
.elite-sub {{
    color: {TEXT_MUTED};
    font-size: 0.8rem;
    margin-top: 4px;
}}
.elite-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.74rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 3px 10px;
    border: 1px solid rgba(255,255,255,0.15);
}}
.elite-chip-positive {{
    color: {POSITIVE};
    border-color: rgba(0,255,136,0.35);
    background: rgba(0,255,136,0.08);
}}
.elite-chip-warning {{
    color: {WARNING};
    border-color: rgba(255,209,102,0.35);
    background: rgba(255,209,102,0.08);
}}
.elite-chip-negative {{
    color: {NEGATIVE};
    border-color: rgba(255,51,102,0.35);
    background: rgba(255,51,102,0.08);
}}
.elite-hero {{
    background: radial-gradient(120% 160% at 0% 0%, rgba(0, 212, 255, 0.08), transparent 40%),
        radial-gradient(120% 180% at 100% 100%, rgba(178, 75, 243, 0.08), transparent 38%),
        rgba(2, 7, 14, 0.96);
    border: 1px solid rgba(0, 212, 255, 0.22);
    border-radius: 14px;
    padding: 16px;
}}
.elite-hero-title {{
    font-family: 'Space Grotesk', 'Manrope', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: {ACCENT};
}}
.elite-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
    margin-top: 10px;
}}
.elite-mini {{
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 8px 10px;
    background: rgba(0,0,0,0.42);
}}

/* Pulse animation for live data */
.pulse {{
    animation: pulse 2s ease infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.6; }}
}}

/* Risk level indicators */
.risk-low {{ color: {POSITIVE}; text-shadow: 0 0 8px rgba(0, 255, 136, 0.4); }}
.risk-medium {{ color: {WARNING}; text-shadow: 0 0 8px rgba(255, 209, 102, 0.4); }}
.risk-high {{ color: {NEGATIVE}; text-shadow: 0 0 8px rgba(255, 51, 102, 0.4); }}
.risk-extreme {{ color: #FF0000; text-shadow: 0 0 12px rgba(255, 0, 0, 0.6); animation: pulse 1s ease infinite; }}

/* Fibonacci level bars */
.fib-level {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 12px;
    margin: 2px 0;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.2s;
}}
.fib-level:hover {{ transform: translateX(4px); }}

/* Monte Carlo probability band */
.mc-stat {{
    text-align: center;
    padding: 12px;
    background: rgba(0, 0, 0, 0.7);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}}

/* Whale tracker entry */
.whale-entry {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: rgba(0, 0, 0, 0.7);
    border-radius: 10px;
    border-left: 3px solid {NEON_BLUE};
    margin: 6px 0;
    transition: all 0.2s;
}}
.whale-entry:hover {{
    background: rgba(0, 0, 0, 0.9);
    border-left-color: {NEON_PURPLE};
}}

/* Tooltip question mark */
.tt {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px; height: 16px;
    border-radius: 50%;
    background: rgba(0, 212, 255, 0.15);
    color: {NEON_BLUE};
    font-size: 0.65rem;
    font-weight: 700;
    cursor: help;
    position: relative;
    vertical-align: middle;
    margin-left: 4px;
    border: 1px solid rgba(0, 212, 255, 0.3);
}}
.tt .ttt {{
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    background: #000000;
    color: {TEXT_LIGHT};
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 400;
    line-height: 1.4;
    width: max-content;
    max-width: 280px;
    white-space: normal;
    z-index: 999;
    border: 1px solid rgba(0, 212, 255, 0.2);
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    transition: opacity 0.2s;
    pointer-events: none;
}}
.tt:hover .ttt {{
    visibility: visible;
    opacity: 1;
}}

/* Mobile-first polish */
@media (max-width: 900px) {{
    .app-page-title--hero {{
        font-size: 2rem !important;
        line-height: 1.15;
    }}
    h1.title {{
        font-size: 2rem;
        line-height: 1.15;
    }}
    h1, .stMarkdown h1 {{
        font-size: 1.9rem !important;
    }}
    h2, .stMarkdown h2 {{
        font-size: 1.35rem !important;
    }}
    h3, .stMarkdown h3 {{
        font-size: 1.08rem !important;
    }}
    .panel-box {{
        padding: 16px;
        margin-bottom: 18px;
        border-radius: 14px;
    }}
    .metric-card {{
        padding: 14px 12px;
        border-radius: 12px;
        margin-bottom: 10px;
    }}
    .metric-label {{
        font-size: 0.68rem;
        letter-spacing: 1px;
    }}
    .metric-value {{
        font-size: 1.25rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.72rem;
        padding: 4px 7px;
    }}
    .app-kpi-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
}}

@media (max-width: 640px) {{
    .stDataFrame, div[data-testid="stDataFrame"] {{
        font-size: 0.78rem !important;
    }}
    .stButton > button {{
        font-size: 0.82rem;
        min-height: 2.7rem;
    }}
    .app-kpi-grid {{
        grid-template-columns: 1fr;
    }}
}}
</style>
"""
