"""
Genomic_UI.py — Professional GUI for Genomic Gene Danger Score Visualization
==============================================================================
Displays AI-ranked gene danger scores from ui_top_10_chart.json and
ui_top_100_table.json in a professional clinical interface.

Uses CustomTkinter for a polished, clinical-grade appearance.
"""

import customtkinter as ctk
import tkinter as tki
import pandas as pd
import json
import os
import sys

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_JSON_PATH = os.path.join(SCRIPT_DIR, 'ui_top_10_chart.json')
TABLE_JSON_PATH = os.path.join(SCRIPT_DIR, 'ui_top_100_table.json')

# ── Clinical colour palette ────────────────────────────────────────────────
PRIMARY_BLUE   = "#1A6B8A"
ACCENT_TEAL    = "#2E9CAD"
LIGHT_BG       = "#EEF2F7"
CARD_BG        = "#FFFFFF"
RESULT_BG      = "#1E2A38"
TEXT_DARK      = "#1A1A2E"
TEXT_MEDIUM    = "#4A5568"
TEXT_LIGHT     = "#FFFFFF"
SUCCESS_GREEN  = "#38A169"
WARNING_YELLOW = "#D69E2E"
ALERT_ORANGE   = "#DD6B20"
CRITICAL_RED   = "#E53E3E"
BAR_BG         = "#E2E8F0"

# ── Danger score → colour gradient ─────────────────────────────────────────
def score_color(score):
    """Return a colour from green→yellow→orange→red based on danger score."""
    if score >= 0.90:
        return CRITICAL_RED
    elif score >= 0.75:
        return ALERT_ORANGE
    elif score >= 0.55:
        return WARNING_YELLOW
    return SUCCESS_GREEN  # Fixed: Added baseline fallback to prevent None crash


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_json(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        print("   Please run genomic1.py first to generate the JSON data.")
        sys.exit(1)
    with open(path, 'r') as f:
        return pd.DataFrame(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════
# GUI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class GenomicUI(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("Genomic Analysis — AI Gene Danger Score Viewer")
        self.geometry("1280x850")
        self.minsize(1100, 750)
        ctk.set_appearance_mode("light")

        # Load data
        self.top10_df  = load_json(CHART_JSON_PATH)
        self.top100_df = load_json(TABLE_JSON_PATH)

        self._build_layout()
        self._build_header()
        self._build_top10_section()
        self._build_top100_section()
        self._build_footer()

    # ─────────────────────────────────────────────────────────────────────
    # LAYOUT
    # ─────────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.grid_rowconfigure(0, weight=0)   # header
        self.grid_rowconfigure(1, weight=1)   # top-10 chart
        self.grid_rowconfigure(2, weight=1)   # top-100 table
        self.grid_rowconfigure(3, weight=0)   # footer
        self.grid_columnconfigure(0, weight=1)

    # ─────────────────────────────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────────────────────────────

    def _build_header(self):
        header = ctk.CTkFrame(self, height=68, fg_color=PRIMARY_BLUE, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        left_frame = ctk.CTkFrame(header, fg_color="transparent")
        left_frame.pack(side="left", padx=28, pady=0)
        left_frame.place(rely=0.5, anchor="w")

        ctk.CTkLabel(
            left_frame, text="🧬",
            font=ctk.CTkFont(size=26),
            text_color=TEXT_LIGHT,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkLabel(
            left_frame,
            text="Genomic Analysis — AI Gene Danger Scores",
            font=ctk.CTkFont(size=19, weight="bold"),
            text_color=TEXT_LIGHT,
        ).pack(side="left")

        ctk.CTkLabel(
            left_frame,
            text="Driver vs Passenger Gene Classification",
            font=ctk.CTkFont(size=11),
            text_color="#B8E0F0",
        ).pack(side="left", padx=16)

        badge = ctk.CTkFrame(header, fg_color=ACCENT_TEAL, corner_radius=14)
        badge.pack(side="right", padx=24, pady=0)
        badge.place(rely=0.5, anchor="e")

        ctk.CTkLabel(
            badge,
            text="  AI ANALYSIS  ",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_LIGHT,
        ).pack(padx=8, pady=4)

    # ─────────────────────────────────────────────────────────────────────
    # TOP-10 CHART SECTION
    # ─────────────────────────────────────────────────────────────────────

    def _build_top10_section(self):
        """Top-10 horizontal stat-bar chart."""
        section = ctk.CTkFrame(self, fg_color=LIGHT_BG, corner_radius=0)
        section.grid(row=1, column=0, sticky="nsew", padx=16, pady=(16, 8))
        section.grid_rowconfigure(0, weight=0)
        section.grid_rowconfigure(1, weight=1)
        section.grid_columnconfigure(0, weight=1)

        # Section header
        hdr = ctk.CTkFrame(section, fg_color=CARD_BG, height=52, corner_radius=8)
        hdr.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 0))
        hdr.grid_propagate(False)

        ctk.CTkLabel(
            hdr,
            text="  🏆  Top 10 Highest Danger Genes",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=PRIMARY_BLUE,
        ).pack(side="left", padx=20)

        ctk.CTkLabel(
            hdr,
            text=f"{len(self.top10_df)} genes ranked by AI danger profile",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MEDIUM,
        ).pack(side="right", padx=20)

        # Chart card
        chart_card = ctk.CTkFrame(section, fg_color=CARD_BG, corner_radius=8)
        chart_card.grid(row=1, column=0, sticky="nsew", padx=14, pady=10)
        chart_card.grid_rowconfigure(0, weight=1)
        chart_card.grid_columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(
            chart_card, fg_color="transparent", corner_radius=0,
            scrollbar_button_color="#CBD5E1",
            scrollbar_button_hover_color="#94A3B8",
        )
        scroll.grid(row=0, column=0, sticky="nsew", padx=16, pady=12)
        scroll.grid_columnconfigure(0, weight=1)

        self._render_statbars(scroll, self.top10_df)

    # ─────────────────────────────────────────────────────────────────────
    # TOP-100 TABLE SECTION
    # ─────────────────────────────────────────────────────────────────────

    def _build_top100_section(self):
        section = ctk.CTkFrame(self, fg_color=LIGHT_BG, corner_radius=0)
        section.grid(row=2, column=0, sticky="nsew", padx=16, pady=(8, 16))
        section.grid_rowconfigure(0, weight=0)
        section.grid_rowconfigure(1, weight=1)
        section.grid_columnconfigure(0, weight=1)

        # Section header
        hdr = ctk.CTkFrame(section, fg_color=CARD_BG, height=52, corner_radius=8)
        hdr.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 0))
        hdr.grid_propagate(False)

        ctk.CTkLabel(
            hdr,
            text="  📊  Top 100 Gene Rankings",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=PRIMARY_BLUE,
        ).pack(side="left", padx=20)

        ctk.CTkLabel(
            hdr,
            text=f"{len(self.top100_df)} total verified entries",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MEDIUM,
        ).pack(side="right", padx=20)

        # Table card
        table_card = ctk.CTkFrame(section, fg_color=CARD_BG, corner_radius=8)
        table_card.grid(row=1, column=0, sticky="nsew", padx=14, pady=10)
        table_card.grid_rowconfigure(0, weight=1)
        table_card.grid_columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(
            table_card, fg_color="transparent", corner_radius=0,
            scrollbar_button_color="#CBD5E1",
            scrollbar_button_hover_color="#94A3B8",
        )
        scroll.grid(row=0, column=0, sticky="nsew", padx=16, pady=12)
        scroll.grid_columnconfigure(0, weight=1)

        self._render_table(scroll, self.top100_df)

    # ─────────────────────────────────────────────────────────────────────
    # STATBAR RENDERER
    # ─────────────────────────────────────────────────────────────────────

    def _render_statbars(self, parent, df):
        """Render horizontal stat bars directly proportional to absolute danger score."""
        BAR_H      = 16
        RANK_W     = 50
        GENE_W     = 180
        SCORE_W    = 80
        CANVAS_W   = 550  # Total canvas lane space

        for idx, row in df.iterrows():
            rank      = df.index.get_loc(idx) + 1
            gene      = row['Gene Name']
            score     = float(row['Danger_Score'])
            
            # Absolute mathematical scaling: size matches danger score exactly (0.0 to 1.0)
            color     = score_color(score)
            score_str = f"{score:.4f}"

            row_frame = ctk.CTkFrame(parent, fg_color="transparent")
            row_frame.pack(fill="x", pady=4)

            # Rank label
            ctk.CTkLabel(
                row_frame, text=f"#{rank:02d}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=TEXT_MEDIUM,
                width=RANK_W,
                anchor="w"
            ).pack(side="left", padx=(5, 5))

            # Gene Name label
            ctk.CTkLabel(
                row_frame, text=gene,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=TEXT_DARK,
                width=GENE_W,
                anchor="w",
            ).pack(side="left", padx=(0, 10))

            # Canvas execution frame container
            bar_canvas = tki.Canvas(
                row_frame,
                width=CANVAS_W,
                height=BAR_H,
                bg=BAR_BG,
                highlightthickness=0,
                bd=0
            )
            bar_canvas.pack(side="left", fill="x", expand=True, pady=4)

            # Fill proportional bar directly tracking standard normalized score limits
            fill_w = int(CANVAS_W * score)
            if fill_w > 0:
                bar_canvas.create_rectangle(0, 0, fill_w, BAR_H, fill=color, outline=color)

            # Absolute Score display field
            ctk.CTkLabel(
                row_frame, text=score_str,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=color,
                width=SCORE_W,
                anchor="e"
            ).pack(side="right", padx=(10, 5))

    # ─────────────────────────────────────────────────────────────────────
    # TABLE RENDERER
    # ─────────────────────────────────────────────────────────────────────

    def _render_table(self, parent, df):
        """Render a clean tabular view of all genes."""
        COLS = ["Rank", "Gene Designation String", "AI Danger Index Evaluation"]

        # Header row
        hdr_row = ctk.CTkFrame(parent, fg_color=PRIMARY_BLUE, corner_radius=6, height=32)
        hdr_row.pack(fill="x", pady=(0, 6))
        hdr_row.grid_columnconfigure(0, weight=1)
        hdr_row.grid_columnconfigure(1, weight=3)
        hdr_row.grid_columnconfigure(2, weight=1)

        for ci, col in enumerate(COLS):
            ctk.CTkLabel(
                hdr_row, text=col,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=TEXT_LIGHT,
                anchor="w" if ci < 2 else "e",
                padx=16,
            ).grid(row=0, column=ci, sticky="ew")

        # Data rows
        for idx, row in df.iterrows():
            rank  = df.index.get_loc(idx) + 1
            gene  = row['Gene Name']
            score = float(row['Danger_Score'])
            color = score_color(score)

            bg = "#F1F5F9" if rank % 2 == 0 else CARD_BG

            data_row = ctk.CTkFrame(parent, fg_color=bg, corner_radius=4)
            data_row.pack(fill="x", pady=2)
            data_row.grid_columnconfigure(0, weight=1)
            data_row.grid_columnconfigure(1, weight=3)
            data_row.grid_columnconfigure(2, weight=1)

            # Rank field
            ctk.CTkLabel(
                data_row, text=f"Order Match {rank:03d}",
                font=ctk.CTkFont(size=12),
                text_color=TEXT_MEDIUM,
                anchor="w", padx=16,
            ).grid(row=0, column=0, sticky="ew")

            # Gene descriptor
            ctk.CTkLabel(
                data_row, text=gene,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=TEXT_DARK,
                anchor="w", padx=16,
            ).grid(row=0, column=1, sticky="ew")

            # Score tracking frame layout
            score_frame = ctk.CTkFrame(data_row, fg_color="transparent")
            score_frame.grid(row=0, column=2, sticky="e", padx=16)

            dot = ctk.CTkFrame(score_frame, width=10, height=10, fg_color=color, corner_radius=5)
            dot.pack(side="left", padx=(0, 8))
            dot.pack_propagate(False)

            ctk.CTkLabel(
                score_frame, text=f"{score:.4f}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=color,
                anchor="e",
            ).pack(side="left")

    # ─────────────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────────────

    def _build_footer(self):
        footer = ctk.CTkFrame(self, height=34, fg_color="#DDE3EA", corner_radius=0)
        footer.grid(row=3, column=0, sticky="ew")
        footer.grid_propagate(False)

        ctk.CTkLabel(
            footer,
            text="Genomic Analysis System  |  AI Driver vs Passenger Gene Classification  |  XGBoost Processing Engine",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MEDIUM,
        ).pack(side="left", padx=20)

        ctk.CTkLabel(
            footer,
            text="v1.0.2 Stable",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_MEDIUM,
        ).pack(side="right", padx=20)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = GenomicUI()
    app.mainloop()