from __future__ import annotations

import csv
from datetime import datetime
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from models import ProductionProblem
from optimizer import grey_wolf_optimizer, is_problem_feasible



ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

WINDOW_BG = "#070B12"
PANEL_BG = "#0B111A"
CARD_BG = "#111A27"
CARD_BG_2 = "#0E1622"
INPUT_BG = "#08101A"
BORDER = "#213246"
BORDER_LIGHT = "#2F4A64"

TEXT_PRIMARY = "#EAF6FF"
TEXT_SECONDARY = "#9AB8CF"
TEXT_MUTED = "#5E768A"

CYAN = "#00C8FF"
BLUE = "#1388FF"
GREEN = "#00E676"
YELLOW = "#FFD740"
RED = "#FF5252"
PURPLE = "#D580FF"

PLOT_BG = "#070B12"
GRID = "#1B2B3A"


DEFAULT_PRODUCTS = [
    {
        "name": "T-shirt",
        "profit": 500,
        "pollution": 1,
        "time": 1,
        "resources": 2,
        "demand": 40,
    },
    {
        "name": "Pantalon",
        "profit": 800,
        "pollution": 2,
        "time": 2,
        "resources": 3,
        "demand": 30,
    },
    {
        "name": "Hoodie",
        "profit": 1200,
        "pollution": 5,
        "time": 3,
        "resources": 4,
        "demand": 20,
    },
]


WOLF_STYLES = {
    "Alpha": (CYAN, "-"),
    "Beta": (GREEN, "--"),
    "Delta": (YELLOW, "-."),
    "Omega": (TEXT_SECONDARY, ":"),
}


CHARTS = [
    ("Profit", GREEN, "DZD", "profit"),
    ("Pollution", RED, "units", "pollution"),
    ("Time", CYAN, "hours", "time"),
    ("Resources", PURPLE, "units", "resources"),
]


def style_axis(axis, title: str, color: str):
    axis.set_facecolor(PLOT_BG)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=7, length=3)

    axis.set_title(
        title,
        color=color,
        fontsize=10,
        fontweight="bold",
        pad=8,
        fontfamily="monospace",
    )

    axis.grid(
        color=GRID,
        linewidth=0.55,
        linestyle="--",
        alpha=0.85,
    )

    for spine in axis.spines.values():
        spine.set_edgecolor(BORDER_LIGHT)
        spine.set_linewidth(0.8)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


class GWOApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GWO Clothing Production Optimizer")
        self.geometry("1480x900")
        self.minsize(1180, 700)
        self.configure(fg_color=WINDOW_BG)

        self.product_rows: list[dict[str, ctk.CTkEntry]] = []
        self.metric_values: dict[str, ctk.CTkLabel] = {}
        self.quantity_values: dict[str, ctk.CTkLabel] = {}

        self.last_problem: ProductionProblem | None = None
        self.last_result = None

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()


    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self,
            width=370,
            fg_color=PANEL_BG,
            corner_radius=0,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        self.sidebar.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=22, pady=(22, 12))

        ctk.CTkLabel(
            header,
            text="METAHEURISTIC OPTIMIZATION",
            text_color=CYAN,
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="GWO Clothing\nOptimizer",
            text_color=TEXT_PRIMARY,
            justify="left",
            font=ctk.CTkFont(size=30, weight="bold"),
        ).pack(anchor="w", pady=(8, 4))

        ctk.CTkLabel(
            header,
            text="Enter product data, constraints, and algorithm settings.",
            text_color=TEXT_SECONDARY,
            justify="left",
            wraplength=310,
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w")

        self.input_scroll = ctk.CTkScrollableFrame(
            self.sidebar,
            fg_color="transparent",
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=BORDER_LIGHT,
        )
        self.input_scroll.grid(row=1, column=0, sticky="nsew", padx=14)

        self._build_input_panel(self.input_scroll)

        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(10, 18))

        self.status_chip = ctk.CTkLabel(
            footer,
            text="● READY",
            text_color=GREEN,
            fg_color=CARD_BG,
            corner_radius=18,
            height=34,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.status_chip.pack(fill="x")

    def _section_title(self, parent, title: str, subtitle: str | None = None):
        ctk.CTkLabel(
            parent,
            text=title,
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", pady=(18, 4))

        if subtitle:
            ctk.CTkLabel(
                parent,
                text=subtitle,
                text_color=TEXT_MUTED,
                justify="left",
                wraplength=310,
                font=ctk.CTkFont(size=11),
            ).pack(anchor="w", pady=(0, 8))

    def _make_entry(self, parent, value="", width=70):
        entry = ctk.CTkEntry(
            parent,
            width=width,
            height=32,
            corner_radius=9,
            fg_color=INPUT_BG,
            border_color=BORDER,
            border_width=1,
            text_color=TEXT_PRIMARY,
            justify="center",
            font=ctk.CTkFont(size=12),
        )
        entry.insert(0, str(value))
        return entry

    def _build_input_panel(self, parent):
        self._section_title(
            parent,
            "Product data",
            "Edit values or add more products.",
        )

        table_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=18)
        table_card.pack(fill="x", pady=(0, 8))

        self.product_table = ctk.CTkFrame(table_card, fg_color="transparent")
        self.product_table.pack(fill="x", padx=12, pady=12)

        headers = ["Product", "Profit", "Poll.", "Time", "Res.", "Demand"]
        widths = [92, 58, 58, 58, 58, 58]

        for col, (header, width) in enumerate(zip(headers, widths)):
            ctk.CTkLabel(
                self.product_table,
                text=header,
                width=width,
                text_color=TEXT_MUTED,
                font=ctk.CTkFont(size=10, weight="bold"),
            ).grid(row=0, column=col, padx=3, pady=(0, 6))

        for product in DEFAULT_PRODUCTS:
            self._add_product_row(product)

        product_buttons = ctk.CTkFrame(parent, fg_color="transparent")
        product_buttons.pack(fill="x", pady=(4, 8))

        ctk.CTkButton(
            product_buttons,
            text="+ Add",
            command=self._add_product_row,
            height=34,
            corner_radius=12,
            fg_color=CARD_BG,
            hover_color=BORDER_LIGHT,
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))

        ctk.CTkButton(
            product_buttons,
            text="− Remove",
            command=self._remove_product_row,
            height=34,
            corner_radius=12,
            fg_color=CARD_BG,
            hover_color=RED,
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left", expand=True, fill="x", padx=4)

        ctk.CTkButton(
            product_buttons,
            text="Reset",
            command=self._reset_defaults,
            height=34,
            corner_radius=12,
            fg_color=CARD_BG,
            hover_color=BORDER_LIGHT,
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left", expand=True, fill="x", padx=(4, 0))

        self._section_title(
            parent,
            "Constraints",
            "Global limits and objective weights.",
        )

        constraints_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=18)
        constraints_card.pack(fill="x", pady=(0, 8))

        self.max_time_entry = self._labeled_entry(
            constraints_card,
            "Maximum time",
            "160",
        )
        self.max_resources_entry = self._labeled_entry(
            constraints_card,
            "Maximum resources",
            "220",
        )
        self.pollution_weight_entry = self._labeled_entry(
            constraints_card,
            "Pollution weight",
            "50",
        )
        self.penalty_entry = self._labeled_entry(
            constraints_card,
            "Constraint penalty",
            "1000",
        )

        self._section_title(
            parent,
            "Algorithm",
            "Tune the search process.",
        )

        algorithm_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=18)
        algorithm_card.pack(fill="x", pady=(0, 12))

        self.wolves_entry = self._labeled_entry(
            algorithm_card,
            "Wolves",
            "30",
        )
        self.iterations_entry = self._labeled_entry(
            algorithm_card,
            "Iterations",
            "100",
        )
        self.seed_entry = self._labeled_entry(
            algorithm_card,
            "Random seed",
            "42",
        )

        self.run_button = ctk.CTkButton(
            parent,
            text="▶ RUN OPTIMIZATION",
            command=self._on_run,
            height=50,
            corner_radius=18,
            fg_color=CYAN,
            hover_color=BLUE,
            text_color="#001018",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.run_button.pack(fill="x", pady=(10, 8))

        export_row = ctk.CTkFrame(parent, fg_color="transparent")
        export_row.pack(fill="x", pady=(0, 18))

        ctk.CTkButton(
            export_row,
            text="Export result",
            command=self._export_results,
            height=38,
            corner_radius=14,
            fg_color=CARD_BG,
            hover_color=BORDER_LIGHT,
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        ctk.CTkButton(
            export_row,
            text="Export input",
            command=self._export_inputs,
            height=38,
            corner_radius=14,
            fg_color=CARD_BG,
            hover_color=BORDER_LIGHT,
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _labeled_entry(self, parent, label: str, default: str):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=7)

        ctk.CTkLabel(
            row,
            text=label,
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=12),
        ).pack(side="left")

        entry = self._make_entry(row, default, width=104)
        entry.pack(side="right")

        return entry

    def _add_product_row(self, product=None):
        row_index = len(self.product_rows) + 1

        product = product or {
            "name": f"Product {row_index}",
            "profit": 500,
            "pollution": 1,
            "time": 1,
            "resources": 1,
            "demand": 10,
        }

        keys = ["name", "profit", "pollution", "time", "resources", "demand"]
        widths = [92, 58, 58, 58, 58, 58]

        row = {}

        for col, (key, width) in enumerate(zip(keys, widths)):
            entry = self._make_entry(self.product_table, product[key], width=width)
            entry.grid(row=row_index, column=col, padx=3, pady=4)
            row[key] = entry

        self.product_rows.append(row)

    def _remove_product_row(self):
        if len(self.product_rows) <= 1:
            messagebox.showwarning(
                "Cannot remove product",
                "At least one product is required.",
            )
            return

        row = self.product_rows.pop()

        for widget in row.values():
            widget.destroy()

    def _reset_defaults(self):
        while self.product_rows:
            row = self.product_rows.pop()

            for widget in row.values():
                widget.destroy()

        for product in DEFAULT_PRODUCTS:
            self._add_product_row(product)

        defaults = [
            (self.max_time_entry, "160"),
            (self.max_resources_entry, "220"),
            (self.pollution_weight_entry, "50"),
            (self.penalty_entry, "1000"),
            (self.wolves_entry, "30"),
            (self.iterations_entry, "100"),
            (self.seed_entry, "42"),
        ]

        for entry, value in defaults:
            entry.delete(0, "end")
            entry.insert(0, value)

        self._set_status("● READY", GREEN)


    def _build_main(self):
        self.main_container = ctk.CTkScrollableFrame(
            self,
            fg_color=WINDOW_BG,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=BORDER_LIGHT,
        )
        self.main_container.grid(row=0, column=1, sticky="nsew")

        self.main_container.grid_columnconfigure(0, weight=1)

        self._build_topbar()
        self._build_results()
        self._build_charts()

    def _build_topbar(self):
        top = ctk.CTkFrame(self.main_container, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 8))
        top.grid_columnconfigure(0, weight=1)

        left = ctk.CTkFrame(top, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            left,
            text="Production Planning Dashboard",
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(anchor="w")

        ctk.CTkLabel(
            left,
            text="Compare scenarios, inspect resource pressure, and export your optimized plan.",
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", pady=(2, 0))

        right = ctk.CTkFrame(top, fg_color=CARD_BG, corner_radius=20)
        right.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            right,
            text="GWO",
            text_color=CYAN,
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(side="left", padx=(18, 4), pady=12)

        ctk.CTkLabel(
            right,
            text="Alpha • Beta • Delta • Omega",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(0, 18), pady=12)

    def _build_results(self):
        results = ctk.CTkFrame(self.main_container, fg_color="transparent")
        results.grid(row=1, column=0, sticky="ew", padx=24, pady=(4, 8))
        results.grid_columnconfigure(0, weight=1)

        self.quantity_panel = ctk.CTkFrame(
            results,
            fg_color=CARD_BG,
            corner_radius=22,
        )
        self.quantity_panel.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        ctk.CTkLabel(
            self.quantity_panel,
            text="Recommended Production Quantities",
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(16, 4))

        self.quantity_cards = ctk.CTkScrollableFrame(
            self.quantity_panel,
            height=82,
            orientation="horizontal",
            fg_color="transparent",
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=BORDER_LIGHT,
        )
        self.quantity_cards.pack(fill="x", padx=14, pady=(4, 14))

        self._refresh_quantity_cards([])

        metric_panel = ctk.CTkFrame(results, fg_color="transparent")
        metric_panel.grid(row=0, column=1, sticky="e")

        for index, (title, key, color) in enumerate(
            [
                ("Profit", "profit", GREEN),
                ("Pollution", "pollution", RED),
                ("Time", "time", CYAN),
                ("Resources", "resources", PURPLE),
                ("Fitness", "fitness", YELLOW),
            ]
        ):
            card = self._metric_card(metric_panel, title, key, color)
            card.grid(row=0, column=index, padx=5)

    def _metric_card(self, parent, title: str, key: str, color: str):
        card = ctk.CTkFrame(
            parent,
            fg_color=CARD_BG,
            corner_radius=20,
            width=112,
            height=110,
        )
        card.grid_propagate(False)

        ctk.CTkLabel(
            card,
            text=title.upper(),
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=10, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(14, 3))

        value = ctk.CTkLabel(
            card,
            text="—",
            text_color=color,
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        value.pack(anchor="w", padx=14)

        self.metric_values[key] = value

        ctk.CTkLabel(
            card,
            text="latest run",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w", padx=14, pady=(0, 6))

        return card

    def _refresh_quantity_cards(self, products: list[str]):
        for widget in self.quantity_cards.winfo_children():
            widget.destroy()

        self.quantity_values = {}

        if not products:
            ctk.CTkLabel(
                self.quantity_cards,
                text="Run the optimizer to generate a production plan.",
                text_color=TEXT_MUTED,
                font=ctk.CTkFont(size=13),
            ).pack(anchor="w", padx=6, pady=24)
            return

        for product in products:
            card = ctk.CTkFrame(
                self.quantity_cards,
                fg_color=CARD_BG_2,
                corner_radius=18,
                width=136,
                height=72,
            )
            card.pack(side="left", padx=5, pady=4)
            card.pack_propagate(False)

            ctk.CTkLabel(
                card,
                text=product.upper()[:16],
                text_color=TEXT_MUTED,
                font=ctk.CTkFont(size=10, weight="bold"),
            ).pack(anchor="w", padx=14, pady=(8, 0))

            value = ctk.CTkLabel(
                card,
                text="—",
                text_color=CYAN,
                font=ctk.CTkFont(size=24, weight="bold"),
            )
            value.pack(anchor="w", padx=14)

            self.quantity_values[product] = value

    def _build_charts(self):
        chart_card = ctk.CTkFrame(
            self.main_container,
            fg_color=CARD_BG,
            corner_radius=24,
        )
        chart_card.grid(row=2, column=0, sticky="ew", padx=24, pady=(8, 24))
        chart_card.grid_columnconfigure(0, weight=1)

        chart_header = ctk.CTkFrame(chart_card, fg_color="transparent")
        chart_header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 4))
        chart_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            chart_header,
            text="Optimization Trajectory",
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            chart_header,
            text="Alpha is the best candidate solution at each iteration.",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=11),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.figure = plt.Figure(figsize=(9.6, 6.6), dpi=100)
        self.figure.patch.set_facecolor(CARD_BG)

        grid = gridspec.GridSpec(
            2,
            2,
            figure=self.figure,
            hspace=0.45,
            wspace=0.30,
            left=0.07,
            right=0.98,
            top=0.94,
            bottom=0.10,
        )

        self.axes = [
            self.figure.add_subplot(grid[row, col])
            for row in range(2)
            for col in range(2)
        ]

        for axis, (title, color, _, _) in zip(self.axes, CHARTS):
            style_axis(axis, title, color)
            axis.text(
                0.5,
                0.5,
                "press RUN to begin",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=TEXT_MUTED,
                fontfamily="monospace",
            )

        canvas_frame = ctk.CTkFrame(chart_card, fg_color="transparent")
        canvas_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))

        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=CARD_BG, highlightthickness=0)
        widget.pack(fill="both", expand=True)


    def _read_problem(self) -> ProductionProblem:
        products = []
        profits = []
        pollution = []
        time = []
        resources = []
        demand = []

        for row in self.product_rows:
            name = row["name"].get().strip()

            if not name:
                raise ValueError("Product names cannot be empty.")

            products.append(name)
            profits.append(float(row["profit"].get()))
            pollution.append(float(row["pollution"].get()))
            time.append(float(row["time"].get()))
            resources.append(float(row["resources"].get()))
            demand.append(float(row["demand"].get()))

        numeric_groups = [profits, pollution, time, resources, demand]

        if any(any(value < 0 for value in group) for group in numeric_groups):
            raise ValueError("Product values must be positive or zero.")

        if any(value <= 0 for value in demand):
            raise ValueError("Demand must be greater than zero for every product.")

        max_time = float(self.max_time_entry.get())
        max_resources = float(self.max_resources_entry.get())
        pollution_weight = float(self.pollution_weight_entry.get())
        penalty = float(self.penalty_entry.get())

        if max_time <= 0 or max_resources <= 0:
            raise ValueError("Maximum time and maximum resources must be greater than zero.")

        if pollution_weight < 0:
            raise ValueError("Pollution weight must be positive or zero.")

        if penalty <= 0:
            raise ValueError("Constraint penalty must be greater than zero.")

        return ProductionProblem(
            products=products,
            profits=np.array(profits, dtype=float),
            pollution=np.array(pollution, dtype=float),
            time=np.array(time, dtype=float),
            resources=np.array(resources, dtype=float),
            demand=np.array(demand, dtype=float),
            max_time=max_time,
            max_resources=max_resources,
            pollution_weight=pollution_weight,
            constraint_penalty=penalty,
        )

    def _read_algorithm_settings(self):
        wolves = int(self.wolves_entry.get())
        iterations = int(self.iterations_entry.get())
        seed = int(self.seed_entry.get())

        if wolves < 4:
            raise ValueError("Number of wolves must be at least 4.")

        if iterations < 1:
            raise ValueError("Number of iterations must be at least 1.")

        return wolves, iterations, seed


    def _on_run(self):
        try:
            problem = self._read_problem()
            wolves, iterations, seed = self._read_algorithm_settings()

            if not is_problem_feasible(problem):
                messagebox.showwarning(
                    "Possibly infeasible input",
                    "No single product appears producible within the current time/resource limits. "
                    "The optimizer will still run, but the result may be heavily penalized.",
                )

        except ValueError as error:
            messagebox.showerror("Invalid input", str(error))
            return

        self.run_button.configure(state="disabled", text="OPTIMIZING...")
        self._set_status("● OPTIMIZING", YELLOW)

        self.after(
            40,
            lambda: self._execute(problem, wolves, iterations, seed),
        )

    def _execute(self, problem, wolves, iterations, seed):
        try:
            result = grey_wolf_optimizer(
                problem=problem,
                num_wolves=wolves,
                max_iter=iterations,
                seed=seed,
            )

            self.last_problem = problem
            self.last_result = result

            self._render_result(problem, result)
            self._plot_histories(result.histories)

            if all(result.constraint_status.values()):
                self._set_status("● DONE", GREEN)
            else:
                self._set_status("● DONE", YELLOW)

        except Exception as error:
            messagebox.showerror("Optimization error", str(error))
            self._set_status("● ERROR", RED)

        finally:
            self.run_button.configure(
                state="normal",
                text="▶ RUN OPTIMIZATION",
            )

    def _render_result(self, problem, result):
        self._refresh_quantity_cards(problem.products)

        for product, quantity in zip(problem.products, result.solution):
            self.quantity_values[product].configure(text=str(int(quantity)))

        self.metric_values["profit"].configure(text=f"{int(result.profit):,}")
        self.metric_values["pollution"].configure(text=str(int(result.pollution)))
        self.metric_values["time"].configure(text=str(int(result.time)))
        self.metric_values["resources"].configure(text=str(int(result.resources)))
        self.metric_values["fitness"].configure(text=f"{result.fitness:.1f}")

    def _plot_histories(self, histories):
        for axis, (title, color, unit, key) in zip(self.axes, CHARTS):
            axis.clear()
            style_axis(axis, title, color)

            axis.set_xlabel(
                "Iteration",
                color=TEXT_SECONDARY,
                fontsize=7.5,
                labelpad=3,
                fontfamily="monospace",
            )

            axis.set_ylabel(
                unit,
                color=TEXT_SECONDARY,
                fontsize=7.5,
                labelpad=3,
                fontfamily="monospace",
            )

            series = histories[key]
            iterations = range(1, len(series["Alpha"]) + 1)

            for role, (line_color, style) in WOLF_STYLES.items():
                values = series[role]

                axis.plot(
                    iterations,
                    values,
                    color=line_color,
                    linewidth=2.3 if role == "Alpha" else 1.35,
                    linestyle=style,
                    alpha=1.0 if role == "Alpha" else 0.72,
                    label=role,
                )

                if role == "Alpha" and values:
                    axis.plot(
                        list(iterations)[-1],
                        values[-1],
                        "o",
                        color=line_color,
                        markersize=5,
                        zorder=5,
                    )

            legend = axis.legend(
                fontsize=7,
                ncol=2,
                facecolor=CARD_BG_2,
                edgecolor=BORDER_LIGHT,
                labelcolor=TEXT_SECONDARY,
                loc="best",
                framealpha=0.92,
            )

            for text in legend.get_texts():
                text.set_fontfamily("monospace")

        self.figure.canvas.draw_idle()

    def _set_status(self, text: str, color: str):
        self.status_chip.configure(text=text, text_color=color)

    # ============================================================
    # EXPORT
    # ============================================================

    def _export_results(self):
        if self.last_problem is None or self.last_result is None:
            messagebox.showwarning(
                "No result",
                "Run the optimizer before exporting results.",
            )
            return

        default_name = f"gwo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")],
        )

        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow(["Product", "Recommended Quantity"])

            for product, quantity in zip(
                self.last_problem.products,
                self.last_result.solution,
            ):
                writer.writerow([product, int(quantity)])

            writer.writerow([])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Profit", self.last_result.profit])
            writer.writerow(["Pollution", self.last_result.pollution])
            writer.writerow(["Time", self.last_result.time])
            writer.writerow(["Resources", self.last_result.resources])
            writer.writerow(["Fitness", self.last_result.fitness])

            writer.writerow([])
            writer.writerow(["Constraint", "Satisfied"])

            for key, value in self.last_result.constraint_status.items():
                writer.writerow([key, value])

        messagebox.showinfo(
            "Export complete",
            f"Results exported to:\n{path}",
        )

    def _export_inputs(self):
        try:
            problem = self._read_problem()

        except ValueError as error:
            messagebox.showerror("Invalid input", str(error))
            return

        default_name = f"gwo_inputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")],
        )

        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "Product",
                    "Profit",
                    "Pollution",
                    "Time",
                    "Resources",
                    "Demand",
                ]
            )

            for i, product in enumerate(problem.products):
                writer.writerow(
                    [
                        product,
                        problem.profits[i],
                        problem.pollution[i],
                        problem.time[i],
                        problem.resources[i],
                        problem.demand[i],
                    ]
                )

            writer.writerow([])
            writer.writerow(["Max time", problem.max_time])
            writer.writerow(["Max resources", problem.max_resources])
            writer.writerow(["Pollution weight", problem.pollution_weight])
            writer.writerow(["Constraint penalty", problem.constraint_penalty])

        messagebox.showinfo(
            "Export complete",
            f"Inputs exported to:\n{path}",
        )