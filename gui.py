import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
import threading

matplotlib.use("TkAgg")

from model_utils import HousingEvaluatorBackend, FEATURE_MAP

class HousingEvaluatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Housing Price Evaluator")
        self.root.geometry("1100x850")

        self.backend = HousingEvaluatorBackend()
        self.backend.train_model(train_fraction=0.8)

        self.test_history = []
        self.prompt_version = 1
        self.current_prompt = None
        self.version_history = {}

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.setup_styles()
        self.create_widgets()
        self.load_houses_list()

    def setup_styles(self):
        self.bg_color = "#1e2530"
        self.fg_color = "#f3f4f6"
        self.accent_color = "#4f46e5"
        self.card_bg = "#273142"

        self.root.configure(bg=self.bg_color)

        self.style.configure('.', background=self.bg_color, foreground=self.fg_color)
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('Card.TFrame', background=self.card_bg, relief='flat')
        self.style.configure('TLabel', background=self.bg_color, foreground=self.fg_color, font=('Helvetica', 10))
        self.style.configure('Title.TLabel', font=('Helvetica', 13, 'bold'), foreground="#38bdf8")
        self.style.configure('Header.TLabel', font=('Helvetica', 11, 'bold'), background=self.card_bg, foreground="#38bdf8")
        self.style.configure('CardText.TLabel', background=self.card_bg, foreground=self.fg_color)

        self.style.configure('TButton', font=('Helvetica', 9, 'bold'), background=self.accent_color, foreground="#ffffff")
        self.style.map('TButton', background=[('active', '#4338ca')])

        self.style.configure('TNotebook', background=self.bg_color, borderwidth=0)
        self.style.configure('TNotebook.Tab', background=self.card_bg, foreground=self.fg_color, padding=[10, 4], font=('Helvetica', 9))
        self.style.map('TNotebook.Tab', background=[('selected', self.accent_color)], foreground=[('selected', '#ffffff')])

        # Styling for Combobox and Spinbox to fit the dark theme
        self.style.configure('TCombobox', fieldbackground='#1e2530', background='#273142', foreground='#f3f4f6', arrowcolor='#f3f4f6', bordercolor='#4f46e5', lightcolor='#4f46e5', darkcolor='#4f46e5')
        self.style.map('TCombobox', fieldbackground=[('readonly', '#1e2530')], foreground=[('readonly', '#f3f4f6')])
        
        self.style.configure('TSpinbox', fieldbackground='#1e2530', background='#273142', foreground='#f3f4f6', arrowcolor='#f3f4f6', bordercolor='#4f46e5', lightcolor='#4f46e5', darkcolor='#4f46e5')
        self.style.map('TSpinbox', fieldbackground=[('readonly', '#1e2530')], foreground=[('readonly', '#f3f4f6')])

        self.root.option_add('*TCombobox*Listbox.background', '#1e2530')
        self.root.option_add('*TCombobox*Listbox.foreground', '#f3f4f6')
        self.root.option_add('*TCombobox*Listbox.selectBackground', '#4f46e5')
        self.root.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')

    def create_widgets(self):
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=15, pady=5)

        title_label = ttk.Label(header_frame, text="Housing Price Evaluator", style="Title.TLabel")
        title_label.pack(anchor="w")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=10)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text="Individual Prediction & Analysis")
        self.notebook.add(self.tab4, text="BLEU/ROUGE - LLM Text Quality")

        self.setup_tab1()
        self.setup_tab4()

    # ----------------- TAB 1 -----------------
    def setup_tab1(self):
        model_frame = ttk.Frame(self.tab1, style="Card.TFrame")
        model_frame.pack(fill="x", padx=10, pady=5)

        inner_model = ttk.Frame(model_frame, style="Card.TFrame")
        inner_model.pack(fill="both", padx=10, pady=8)

        ttk.Label(inner_model, text="Regression Model Configuration", style="Header.TLabel").grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 5))

        ttk.Label(inner_model, text="Training data percentage (%):", style="CardText.TLabel").grid(row=1, column=0, sticky="w")
        self.spin_train_frac = ttk.Spinbox(inner_model, from_=10, to=100, increment=10, width=5)
        self.spin_train_frac.set(80)
        self.spin_train_frac.grid(row=1, column=1, sticky="w", padx=5)

        btn_train = ttk.Button(inner_model, text="Train Model", command=self.train_on_fraction)
        btn_train.grid(row=1, column=2, padx=15)

        self.lbl_train_stats = ttk.Label(inner_model, text="", style="CardText.TLabel")
        self.lbl_train_stats.grid(row=1, column=3, columnspan=2, sticky="w")
        self.update_train_stats_label()

        analysis_frame = ttk.Frame(self.tab1, style="Card.TFrame")
        analysis_frame.pack(fill="both", expand=True, padx=10, pady=5)

        inner_analysis = ttk.Frame(analysis_frame, style="Card.TFrame")
        inner_analysis.pack(fill="both", expand=True, padx=10, pady=8)

        ttk.Label(inner_analysis, text="Individual Property Analysis & LLM Alignment", style="Header.TLabel").grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 5))

        ttk.Label(inner_analysis, text="Select property:", style="CardText.TLabel").grid(row=1, column=0, sticky="w")
        self.combo_houses = ttk.Combobox(inner_analysis, width=35, state="readonly")
        self.combo_houses.grid(row=1, column=1, sticky="w", padx=5)
        self.combo_houses.bind("<<ComboboxSelected>>", self.on_house_selected)

        self.btn_explain = ttk.Button(inner_analysis, text="Estimate & Explain", command=self.explain_individual_house)
        self.btn_explain.grid(row=1, column=2, padx=10)

        self.btn_optimize = ttk.Button(inner_analysis, text="Optimization Pipeline", command=self.launch_optimization_pipeline)
        self.btn_optimize.grid(row=1, column=3, padx=10)

        self.force_sim_var = tk.BooleanVar(value=False)
        chk_sim = ttk.Checkbutton(inner_analysis, text="Force simulation", variable=self.force_sim_var)
        chk_sim.grid(row=1, column=4, sticky="w", padx=5)

        self.include_judge_var = tk.BooleanVar(value=True)
        chk_judge = ttk.Checkbutton(inner_analysis, text="Include Judge", variable=self.include_judge_var)
        chk_judge.grid(row=1, column=5, sticky="w", padx=5)

        content_frame = ttk.Frame(inner_analysis, style="Card.TFrame")
        content_frame.grid(row=2, column=0, columnspan=5, sticky="nsew", pady=10)
        inner_analysis.rowconfigure(2, weight=1)
        inner_analysis.columnconfigure(5, weight=1)

        left_subframe = ttk.Frame(content_frame, style="Card.TFrame", width=320)
        left_subframe.pack(side="left", fill="both", expand=False)
        left_subframe.pack_propagate(False)

        self.txt_house_details = tk.Text(left_subframe, height=7, width=40, bg="#1e2530", fg="#f3f4f6", relief="flat", font=('Courier', 9))
        self.txt_house_details.pack(fill="x", pady=5)

        self.txt_price_comparison = tk.Text(left_subframe, height=5, width=40, bg="#1e2530", fg="#38bdf8", relief="flat", font=('Courier', 9))
        self.txt_price_comparison.pack(fill="x", pady=5)

        prompt_frame = ttk.LabelFrame(left_subframe, text="Prompt Version", style="Card.TFrame")
        prompt_frame.pack(fill="x", pady=5)

        self.lbl_prompt_version = ttk.Label(prompt_frame, text="v1 (Original)", font=('Helvetica', 10, 'bold'), foreground="#60a5fa")
        self.lbl_prompt_version.pack(anchor="w", padx=5, pady=5)

        self.plot_frame = ttk.Frame(content_frame, style="Card.TFrame")
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        self.fig_weights, self.ax_weights = plt.subplots(figsize=(5, 3))
        self.fig_weights.patch.set_facecolor('#273142')
        self.ax_weights.set_facecolor('#1e2530')
        self.canvas_weights = FigureCanvasTkAgg(self.fig_weights, master=self.plot_frame)
        self.canvas_weights.get_tk_widget().pack(fill="both", expand=True)

        ttk.Label(inner_analysis, text="Current Prompt:", style="CardText.TLabel").grid(row=3, column=0, columnspan=5, sticky="w", pady=(5, 2))
        self.txt_prompt_display = scrolledtext.ScrolledText(inner_analysis, height=3, width=105, bg="#0f1419", fg="#a0aec0", relief="flat", font=('Courier', 8), wrap=tk.WORD)
        self.txt_prompt_display.grid(row=4, column=0, columnspan=5, sticky="w", pady=5)

        v1_prompt = """PROPERTY VALUATION ANALYSIS

HOUSE DATA:
- Median Income Area: ${MedInc*10000}/year
- House Age: {HouseAge} years
- Average Rooms: {AveRooms}
- Bedrooms: {AveBedrms}
- Population: {Population}
- Occupancy: {AveOccup}
- Location: {Latitude}N, {Longitude}W

ANALYSIS & VALUATION:
Provide 4-5 sentence analysis covering:
1. Income impact on value
2. House age/condition role
3. Room count/space quality
4. Location factors
5. Overall positioning & price

RESPONSE FORMAT:
[4-5 sentences analysis]

Factor Importance Weights:
[MEDINC: ##] [HOUSEAGE: ##] [AVEROOMS: ##] [AVEBEDRMS: ##] [POPULATION: ##] [AVEOCCUP: ##] [LATITUDE: ##] [LONGITUDE: ##]

[PREDICTION: $price]"""

        self.txt_prompt_display.insert(tk.END, v1_prompt)
        self.txt_prompt_display.config(state="disabled")

        ttk.Label(inner_analysis, text="LLM Response:", style="CardText.TLabel").grid(row=5, column=0, columnspan=5, sticky="w", pady=(5, 2))
        self.txt_explanation = scrolledtext.ScrolledText(inner_analysis, height=4, width=105, bg="#1e2530", fg="#f3f4f6", relief="flat", font=('Helvetica', 9, 'italic'), wrap=tk.WORD)
        self.txt_explanation.grid(row=6, column=0, columnspan=5, sticky="w", pady=5)

        self.lbl_metrics_tab1 = ttk.Label(inner_analysis, text="Consistency: -- | BLEU: -- | ROUGE: -- | Judge: -- | Halluc: -- | Omissions: -- | SHAP Hint: --", style="CardText.TLabel", font=('Helvetica', 8, 'bold'))
        self.lbl_metrics_tab1.grid(row=5, column=0, columnspan=6, sticky="w", pady=5)

    def update_train_stats_label(self):
        r2 = self.backend.metrics["r2"]
        mae = self.backend.metrics["mae"]
        size = self.backend.metrics["train_size"]
        self.lbl_train_stats.config(text=f"R-squared: {r2:.4f} | MAE: ${mae:,.0f} | Number of samples: {size}")

    def train_on_fraction(self):
        try:
            frac = float(self.spin_train_frac.get()) / 100.0
            if frac <= 0.0 or frac > 1.0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Training percentage must be between 10 and 100.")
            return

        self.backend.train_model(train_fraction=frac)
        self.update_train_stats_label()
        self.load_houses_list()
        messagebox.showinfo("Training", "Model has been trained on the selected fraction.")

    def load_houses_list(self):
        self.houses = self.backend.get_test_houses(n=50)
        house_strings = [f"House ID {h['id']} - Actual Price: ${h['actual_price']:,.0f}" for h in self.houses]
        self.combo_houses['values'] = house_strings
        if house_strings:
            self.combo_houses.current(0)
            self.on_house_selected(None)

    def on_house_selected(self, event):
        idx = self.combo_houses.current()
        if idx < 0:
            return

        house = self.houses[idx]
        features = house["features"]

        self.txt_house_details.delete("1.0", tk.END)
        self.txt_house_details.insert(tk.END, "PROPERTY DATA:\n")
        self.txt_house_details.insert(tk.END, f"Median Income:    ${features['MedInc']*10:.1f}k/year\n")
        self.txt_house_details.insert(tk.END, f"House Age:    {features['HouseAge']:.0f} years\n")
        self.txt_house_details.insert(tk.END, f"Number of Rooms:    {features['AveRooms']:.2f}\n")
        self.txt_house_details.insert(tk.END, f"Bedrooms:      {features['AveBedrms']:.2f}\n")
        self.txt_house_details.insert(tk.END, f"Area Population:  {int(features['Population'])}\n")
        self.txt_house_details.insert(tk.END, f"Average Occupancy:   {features['AveOccup']:.2f}\n")
        self.txt_house_details.insert(tk.END, f"Coordinates:      {features['Latitude']:.2f}N, {features['Longitude']:.2f}W")

        self.txt_price_comparison.delete("1.0", tk.END)
        self.txt_price_comparison.insert(tk.END, "PRICE ESTIMATES:\n")
        self.txt_price_comparison.insert(tk.END, f"Actual Price:       ${house['actual_price']:,.0f}\n")
        self.txt_price_comparison.insert(tk.END, f"ML Model Price:   ${house['predicted_price']:,.0f}\n")
        self.txt_price_comparison.insert(tk.END, "LLM Estimate: Not calculated\n")
        self.txt_price_comparison.insert(tk.END, "LLM Price Error:  Not calculated")

        self.txt_explanation.delete("1.0", tk.END)
        self.lbl_metrics_tab1.config(text="Keyword consistency: unevaluated | Weight distribution similarity (L1): unevaluated | LLM estimation error: unevaluated")

        self.ax_weights.clear()
        self.ax_weights.set_title("Feature Weights Comparison (Model vs LLM)", color=self.fg_color, fontsize=9)
        self.fig_weights.tight_layout()
        self.canvas_weights.draw()

    def explain_individual_house(self):
        idx = self.combo_houses.current()
        if idx < 0:
            return

        house = self.houses[idx]
        features = house["features"]
        actual_price = house["actual_price"]

        self.btn_explain.config(state="disabled")
        self.root.update()

        try:
            force_sim = self.force_sim_var.get()
            include_judge = self.include_judge_var.get()
            use_v2 = (self.prompt_version >= 2)

            # ---- SINGLE PIPELINE CALL ----
            result = self.backend.run_full_pipeline(
                features, actual_price,
                force_simulation=force_sim,
                use_v2=use_v2,
                include_judge=include_judge
            )

            pred_price = result["pred_price"]
            explanation_text = result["explanation"]
            llm_pred = result["llm_prediction"]
            contributions = result["contributions"]
            bleu = result["bleu"]
            rouge = result["rouge"]
            was_simulated = result["was_simulated"]

            # ---- UPDATE LLM RESPONSE ----
            self.txt_explanation.delete("1.0", tk.END)
            self.txt_explanation.insert(tk.END, explanation_text)

            # ---- UPDATE PRICE COMPARISON ----
            self.txt_price_comparison.delete("1.0", tk.END)
            self.txt_price_comparison.insert(tk.END, "PRICE ESTIMATES:\n")
            self.txt_price_comparison.insert(tk.END, f"Actual Price:       ${actual_price:,.0f}\n")
            self.txt_price_comparison.insert(tk.END, f"ML Model Price:   ${pred_price:,.0f}\n")

            pe = result["llm_price_error_pct"]
            if llm_pred is not None:
                self.txt_price_comparison.insert(tk.END, f"LLM Estimate: ${llm_pred:,.0f}\n")
                self.txt_price_comparison.insert(tk.END, f"LLM Price Error:   {pe:.1f}%\n")
                err_text = f"{pe:.1f}%"
            else:
                self.txt_price_comparison.insert(tk.END, "LLM Estimate: N/A (Error format)\n")
                self.txt_price_comparison.insert(tk.END, "LLM Price Error:   N/A\n")
                err_text = "N/A"

            # ---- UPDATE METRICS LABEL (all pipeline outputs) ----
            cons_pct = result["factual_consistency"] * 100.0
            judge_str = f"{result['judge_score']:.1f}/10" if result.get('judge_score') is not None else "--"
            halluc_count = len(result["hallucinations"])
            omission_count = len(result["omissions"])
            hint = result["shap_hint_strength"]

            self.lbl_metrics_tab1.config(
                text=(
                    f"Consistency: {cons_pct:.1f}% | "
                    f"BLEU: {bleu:.1f}% | ROUGE: {rouge:.1f}% | "
                    f"Judge: {judge_str} | "
                    f"Halluc: {halluc_count} | Omissions: {omission_count} | "
                    f"SHAP Hint: {hint}"
                )
            )

            # ---- UPDATE PLOTS ----
            self.draw_weights_comparison_plot(result["real_weights"], result["llm_weights"])
            self.update_bleu_rouge_dashboard(result)

            # ---- BUILD TEST RECORD (from pipeline result) ----
            test_record = {
                'house_id': house['id'],
                'timestamp': __import__('datetime').datetime.now(),
                'prompt_version': self.prompt_version,
                'actual_price': actual_price,
                'ml_price': pred_price,
                'llm_price': llm_pred if llm_pred else 0,
                'bleu': bleu,
                'rouge': rouge,
                'consistency': cons_pct,
                'price_error_llm': pe if pe is not None else 999,
                'factors_matched': len(set(result["top_real_factors"]) & set(result["keywords_detected"])),
                'top_real_factors': result["top_real_factors"],
                'explanation_text': explanation_text,
                'judge_score': result["judge_score"],
                'judge_reason': result["judge_reason"],
                'was_simulated': was_simulated,
                'hallucinations': result["hallucinations"],
                'omissions': result["omissions"],
                'shap_hint_strength': hint,
            }
            self.test_history.append(test_record)

            if self.prompt_version not in self.version_history:
                self.version_history[self.prompt_version] = {'tests': [], 'metrics': {}}
            self.version_history[self.prompt_version]['tests'].append(test_record)

        except Exception as e:
            messagebox.showerror("Error", f"Could not generate explanation: {str(e)}")
        finally:
            self.btn_explain.config(state="normal")

    def draw_weights_comparison_plot(self, real_weights, llm_weights):
        self.ax_weights.clear()

        labels = [FEATURE_MAP[name]["en"][:10] for name in self.backend.feature_names]
        x = np.arange(len(labels))
        width = 0.35

        y_real = [real_weights[name] for name in self.backend.feature_names]
        y_llm = [llm_weights[name] for name in self.backend.feature_names]

        self.ax_weights.bar(x - width/2, y_real, width, label='RF Model', color='#06b6d4')
        self.ax_weights.bar(x + width/2, y_llm, width, label='LLM Estimate', color='#8b5cf6')

        self.ax_weights.set_ylabel('Importance (%)', color=self.fg_color, fontsize=8)
        self.ax_weights.set_title('Weight Comparison: Decision Model vs LLM Declared', color=self.fg_color, fontsize=9)
        self.ax_weights.set_xticks(x)
        self.ax_weights.set_xticklabels(labels, rotation=35, ha='right', color=self.fg_color, fontsize=7)
        self.ax_weights.tick_params(colors=self.fg_color, labelsize=7)
        self.ax_weights.legend(facecolor='#273142', edgecolor='none', labelcolor=self.fg_color, fontsize=8)

        self.fig_weights.tight_layout()
        self.canvas_weights.draw()

    # --------- TAB 4 ---------
    def setup_tab4(self):
        header_frame = ttk.Frame(self.tab4, style="Card.TFrame")
        header_frame.pack(fill="x", padx=10, pady=5)

        inner_header = ttk.Frame(header_frame, style="Card.TFrame")
        inner_header.pack(fill="both", padx=10, pady=8)

        ttk.Label(inner_header, text="LLM Text Quality Evaluation - BLEU & ROUGE Analysis", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(inner_header, text="Compare ideal vs LLM explanation. Green=ideal matches, Orange=LLM matches.", font=('Helvetica', 9), foreground="#9ca3af").pack(anchor="w")

        content_frame = ttk.Frame(self.tab4, style="Card.TFrame")
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        chart_frame = ttk.Frame(self.tab4, style="Card.TFrame")
        chart_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(chart_frame, text="Word Frequency & Complexity Analysis:", style="Header.TLabel").pack(anchor="w", pady=(5, 5))
        self.chart_display = tk.Text(chart_frame, height=6, width=120, bg="#1e2530", fg="#d1d5db", relief="flat", font=('Courier', 8))
        self.chart_display.pack(fill="both")

        left_frame = ttk.Frame(content_frame, style="Card.TFrame")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ttk.Label(left_frame, text="Ideal Explanation (from real factors):", style="Header.TLabel").pack(anchor="w", pady=(5, 5))
        self.txt_ideal_explanation = scrolledtext.ScrolledText(left_frame, height=6, width=50, bg="#1e2530", fg="#10b981", relief="flat", font=('Helvetica', 9), wrap=tk.WORD)
        self.txt_ideal_explanation.pack(fill="both", expand=True, pady=(0, 10))

        ttk.Label(left_frame, text="LLM Explanation (from Tab 1):", style="Header.TLabel").pack(anchor="w", pady=(5, 5))
        self.txt_llm_explanation_tab4 = scrolledtext.ScrolledText(left_frame, height=6, width=50, bg="#1e2530", fg="#f59e0b", relief="flat", font=('Helvetica', 9), wrap=tk.WORD)
        self.txt_llm_explanation_tab4.pack(fill="both", expand=True)

        right_frame = ttk.Frame(content_frame, style="Card.TFrame", width=280)
        right_frame.pack(side="right", fill="both", expand=False, padx=(10, 0))
        right_frame.pack_propagate(False)

        ttk.Label(right_frame, text="TEXT QUALITY METRICS:", style="Header.TLabel").pack(anchor="w", pady=(5, 10))

        bleu_frame = ttk.Frame(right_frame, style="Card.TFrame")
        bleu_frame.pack(fill="x", pady=5)
        ttk.Label(bleu_frame, text="BLEU Score (N-gram):", style="CardText.TLabel", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        self.lbl_bleu_tab4 = ttk.Label(bleu_frame, text="-- %", font=('Helvetica', 20, 'bold'), foreground="#60a5fa")
        self.lbl_bleu_tab4.pack(anchor="w")
        self.prog_bleu = ttk.Progressbar(bleu_frame, length=200, mode='determinate', maximum=100)
        self.prog_bleu.pack(fill="x", pady=3)
        ttk.Label(bleu_frame, text="Phrase matching (2-word overlap)", style="CardText.TLabel", font=('Helvetica', 8)).pack(anchor="w")

        rouge_frame = ttk.Frame(right_frame, style="Card.TFrame")
        rouge_frame.pack(fill="x", pady=5)
        ttk.Label(rouge_frame, text="ROUGE Score (LCS):", style="CardText.TLabel", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        self.lbl_rouge_tab4 = ttk.Label(rouge_frame, text="-- %", font=('Helvetica', 20, 'bold'), foreground="#f87171")
        self.lbl_rouge_tab4.pack(anchor="w")
        self.prog_rouge = ttk.Progressbar(rouge_frame, length=200, mode='determinate', maximum=100)
        self.prog_rouge.pack(fill="x", pady=3)
        ttk.Label(rouge_frame, text="Semantic overlap (content similarity)", style="CardText.TLabel", font=('Helvetica', 8)).pack(anchor="w")

        quality_frame = ttk.Frame(right_frame, style="Card.TFrame")
        quality_frame.pack(fill="x", pady=5)
        ttk.Label(quality_frame, text="OVERALL QUALITY:", style="Header.TLabel").pack(anchor="w")
        self.lbl_quality_score = ttk.Label(quality_frame, text="-- %", font=('Helvetica', 18, 'bold'), foreground="#8b5cf6")
        self.lbl_quality_score.pack(anchor="w")
        self.prog_quality = ttk.Progressbar(quality_frame, length=200, mode='determinate', maximum=100)
        self.prog_quality.pack(fill="x", pady=3)
        self.lbl_quality_text = ttk.Label(quality_frame, text="", style="CardText.TLabel", font=('Helvetica', 8))
        self.lbl_quality_text.pack(anchor="w")

        interp_frame = ttk.Frame(right_frame, style="Card.TFrame")
        interp_frame.pack(fill="both", expand=True, pady=10)

        ttk.Label(interp_frame, text="DETAILED ANALYSIS:", style="Header.TLabel").pack(anchor="w", pady=(5, 3))
        self.txt_interpretation = scrolledtext.ScrolledText(interp_frame, height=10, width=35, bg="#1e2530", fg="#d1d5db", relief="flat", font=('Helvetica', 7, 'bold'), wrap=tk.WORD)
        self.txt_interpretation.pack(fill="both", expand=True)

        btn_frame = ttk.Frame(right_frame, style="Card.TFrame")
        btn_frame.pack(fill="x", pady=(5, 0))
        ttk.Button(btn_frame, text="Copy", command=self.copy_comparison).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Details", command=self.show_detailed_analysis).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Factors", command=self.show_factor_analysis).pack(side="left", padx=2)

    def update_bleu_rouge_dashboard(self, pipeline_result):
        """Update Tab 4 BLEU/ROUGE dashboard from a pipeline result dict."""
        bleu = pipeline_result["bleu"]
        rouge = pipeline_result["rouge"]
        ideal_explanation = pipeline_result["ideal_explanation"]
        llm_explanation_text = pipeline_result["explanation"]
        contributions = pipeline_result["contributions"]
        top_real_factors = pipeline_result["top_real_factors"]
        factual_consistency = pipeline_result["factual_consistency"] * 100

        llm_text_only = llm_explanation_text.split('[')[0].strip() if '[' in llm_explanation_text else llm_explanation_text[:400]

        self.txt_ideal_explanation.tag_configure("match", background="#10b981", foreground="white", font=('Helvetica', 9, 'bold'))
        self.txt_llm_explanation_tab4.tag_configure("match", background="#f59e0b", foreground="white", font=('Helvetica', 9, 'bold'))

        self.txt_ideal_explanation.delete("1.0", tk.END)
        self.txt_ideal_explanation.insert(tk.END, ideal_explanation)

        self.txt_llm_explanation_tab4.delete("1.0", tk.END)
        self.txt_llm_explanation_tab4.insert(tk.END, llm_text_only)

        ideal_words = ideal_explanation.lower().split()
        llm_words = llm_text_only.lower().split()

        ideal_2grams = set([' '.join(ideal_words[i:i+2]) for i in range(len(ideal_words)-1)])
        llm_2grams = [' '.join(llm_words[i:i+2]) for i in range(len(llm_words)-1)]
        matching_2grams = [ng for ng in llm_2grams if ng in ideal_2grams]

        for i in range(len(ideal_words)-1):
            phrase = ' '.join(ideal_words[i:i+2])
            if phrase in ideal_2grams:
                for match_phrase in matching_2grams:
                    if phrase.lower() == match_phrase.lower():
                        idx = ideal_explanation.lower().find(phrase.lower())
                        if idx >= 0:
                            self.txt_ideal_explanation.tag_add("match", f"1.0+{idx}c", f"1.0+{idx+len(phrase)}c")
                            break

        for i in range(len(llm_words)-1):
            phrase = ' '.join(llm_words[i:i+2])
            if phrase in ideal_2grams:
                idx2 = llm_text_only.lower().find(phrase.lower())
                if idx2 >= 0:
                    self.txt_llm_explanation_tab4.tag_add("match", f"1.0+{idx2}c", f"1.0+{idx2+len(phrase)}c")

        self.lbl_bleu_tab4.config(text=f"{bleu:.1f}%")
        self.lbl_rouge_tab4.config(text=f"{rouge:.1f}%")
        self.prog_bleu['value'] = bleu
        self.prog_rouge['value'] = rouge

        self.current_bleu = bleu
        self.current_rouge = rouge
        self.current_ideal = ideal_explanation
        self.current_llm = llm_text_only
        self.current_contributions = contributions
        self.current_top_factors = top_real_factors

        quality_score = (bleu * 0.3 + rouge * 0.3 + factual_consistency * 0.4)

        self.lbl_quality_score.config(text=f"{quality_score:.1f}%")
        self.prog_quality['value'] = quality_score

        if quality_score >= 80:
            quality_text = "Excellent: LLM explanation well-aligned with ML factors"
        elif quality_score >= 60:
            quality_text = "Good: Moderate alignment, some deviations"
        elif quality_score >= 40:
            quality_text = "Fair: Significant differences between ideal & LLM"
        else:
            quality_text = "Poor: Major factual/conceptual misalignment"

        self.lbl_quality_text.config(text=quality_text)

        ref_tokens = ideal_explanation.lower().split()
        gen_tokens = llm_text_only.lower().split()

        def lcs_length(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        lcs = lcs_length(ref_tokens, gen_tokens)
        ideal_word_count = len(ref_tokens)
        llm_word_count = len(gen_tokens)

        # Compute n-gram match stats for interpretation
        bleu_matches = 0
        bleu_total = max(len(gen_tokens) - 2 + 1, 0)
        if bleu_total > 0:
            from collections import Counter
            ref_ngrams = [' '.join(ref_tokens[i:i+2]) for i in range(len(ref_tokens) - 1)]
            gen_ngrams = [' '.join(gen_tokens[i:i+2]) for i in range(len(gen_tokens) - 1)]
            ref_c = Counter(ref_ngrams)
            gen_c = Counter(gen_ngrams)
            for ng in gen_c:
                bleu_matches += min(gen_c[ng], ref_c.get(ng, 0))

        interp = "TECHNICAL METRICS:\n"
        interp += "="*32 + "\n\n"
        interp += f"BLEU: {bleu:.2f}%\n"
        interp += f"Formula: matches / total_ngrams\n"
        interp += f"Details:\n"
        interp += f"  Matches: {bleu_matches}\n"
        interp += f"  2-grams: {bleu_total}\n"
        interp += f"  = {bleu_matches}/{bleu_total} = {(bleu_matches/bleu_total*100) if bleu_total>0 else 0:.2f}%\n"
        interp += f"Ref: {ideal_word_count} words\n"
        interp += f"Gen: {llm_word_count} words\n"
        interp += f"\nROUGE: {rouge:.2f}%\n"
        interp += f"Formula: LCS / ref_words\n"
        interp += f"Details:\n"
        interp += f"  LCS length: {lcs}\n"
        interp += f"  Ref words: {ideal_word_count}\n"
        interp += f"  = {lcs}/{ideal_word_count} = {(lcs/ideal_word_count*100) if ideal_word_count>0 else 0:.2f}%\n"
        interp += f"Precision: {(lcs/llm_word_count*100) if llm_word_count>0 else 0:.2f}%\n"
        interp += f"\nFactors: {', '.join([FEATURE_MAP[f]['en'][:8] for f in top_real_factors[:2]])}\n"

        # Show judge + hallucinations in interpretation
        judge_s = pipeline_result.get('judge_score')
        if judge_s is not None:
            interp += f"\nJudge: {judge_s:.1f}/10\n"
            interp += f"Reason: {pipeline_result.get('judge_reason', '')}\n"
        interp += f"Hallucinations: {len(pipeline_result['hallucinations'])}\n"
        interp += f"Omissions: {len(pipeline_result['omissions'])}\n"
        interp += f"SHAP Hint: {pipeline_result['shap_hint_strength']}\n"

        self.txt_interpretation.delete("1.0", tk.END)
        self.txt_interpretation.insert(tk.END, interp)

        self.update_complexity_analysis(ideal_explanation, llm_text_only)

    def generate_ideal_explanation(self, features, top_factors, contributions=None):
        ideal = ""
        if top_factors:
            primary_factor = top_factors[0]
            primary_label = FEATURE_MAP[primary_factor]['en']

            if primary_factor == "MedInc":
                income = features['MedInc'] * 10000
                ideal += f"This property's primary value driver is location income level at ${income:,.0f}/year median household income. "
                if income > 50000:
                    ideal += "This high-income area supports strong property valuations. "
                else:
                    ideal += "This moderate income area represents standard property pricing. "
            elif primary_factor == "HouseAge":
                age = features['HouseAge']
                ideal += f"Property age at {age:.0f} years is the dominant valuation factor. "
                if age < 30:
                    ideal += "Relatively modern construction supports higher values. "
                else:
                    ideal += "Older construction affects market valuation. "
            elif primary_factor in ["AveRooms", "AveBedrms"]:
                rooms = features['AveRooms']
                ideal += f"Property size with {rooms:.2f} average rooms is the key value driver. "
                if rooms > 6:
                    ideal += "Larger properties command premium pricing. "
                else:
                    ideal += "Compact properties represent efficient living space. "
            elif primary_factor in ["Latitude", "Longitude"]:
                ideal += f"Geographic location at {features['Latitude']:.2f}N, {features['Longitude']:.2f}W is the primary pricing factor. "
                if 37 <= features['Latitude'] <= 38 and -124 <= features['Longitude'] <= -121:
                    ideal += "Bay Area coastal positioning provides significant premium value. "

        if len(top_factors) > 1:
            ideal += f"Secondary impact from {FEATURE_MAP[top_factors[1]]['en'].lower()}. "
        if len(top_factors) > 2:
            ideal += f"Tertiary influence from {FEATURE_MAP[top_factors[2]]['en'].lower()}. "

        ideal += "Combined analysis of these factors determines the property's market valuation."
        return ideal

    def calculate_bleu_score(self, reference, generated, n=2):
        from collections import Counter
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()

        if len(gen_tokens) == 0:
            return 0.0

        matches = 0
        total = max(len(gen_tokens) - n + 1, 0)

        if total == 0:
            return 0.0

        ref_ngrams = [' '.join(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
        gen_ngrams = [' '.join(gen_tokens[i:i+n]) for i in range(len(gen_tokens) - n + 1)]

        ref_counts = Counter(ref_ngrams)
        gen_counts = Counter(gen_ngrams)

        for ngram in gen_counts:
            matches += min(gen_counts[ngram], ref_counts.get(ngram, 0))

        bleu_score = (matches / total) * 100 if total > 0 else 0.0

        self.bleu_matches = matches
        self.bleu_total = total
        self.bleu_precision = bleu_score

        return bleu_score

    def calculate_rouge_score(self, reference, generated):
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()

        if len(ref_tokens) == 0 or len(gen_tokens) == 0:
            return 0.0

        def lcs_length(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        lcs = lcs_length(ref_tokens, gen_tokens)
        recall = (lcs / len(ref_tokens)) * 100 if len(ref_tokens) > 0 else 0.0
        precision = (lcs / len(gen_tokens)) * 100 if len(gen_tokens) > 0 else 0.0
        return (recall + precision) / 2 if recall > 0 or precision > 0 else 0.0

    def copy_comparison(self):
        comparison = f"BLEU: {self.current_bleu:.2f}%\nROUGE: {self.current_rouge:.2f}%\n\n"
        comparison += f"IDEAL:\n{self.current_ideal}\n\nLLM:\n{self.current_llm}"
        self.root.clipboard_clear()
        self.root.clipboard_append(comparison)
        messagebox.showinfo("Copied", "Comparison data copied to clipboard!")

    def update_complexity_analysis(self, ideal_text, llm_text):
        from collections import Counter
        ideal_words = ideal_text.lower().split()
        llm_words = llm_text.lower().split()

        ideal_freq = Counter(ideal_words)
        llm_freq = Counter(llm_words)

        ideal_unique = len(ideal_freq)
        llm_unique = len(llm_freq)
        ideal_avg_len = sum(len(w) for w in ideal_words) / len(ideal_words) if ideal_words else 0
        llm_avg_len = sum(len(w) for w in llm_words) / len(llm_words) if llm_words else 0

        missing = set(ideal_words) - set(llm_words)
        extra = set(llm_words) - set(ideal_words)

        analysis = ""
        analysis += f"{'IDEAL':<40} | {'LLM':<40}\n"
        analysis += f"{'-'*40} | {'-'*40}\n"
        analysis += f"Words: {len(ideal_words):<34} | Words: {len(llm_words):<34}\n"
        analysis += f"Unique: {ideal_unique:<34} | Unique: {llm_unique:<34}\n"
        analysis += f"Avg word len: {ideal_avg_len:<28.2f} | Avg word len: {llm_avg_len:<28.2f}\n"
        analysis += f"Vocabulary match: {len(set(ideal_words) & set(llm_words)):<25} | Missing words: {len(missing):<25}\n"
        analysis += f"\nMissing (in ideal, not in LLM): {', '.join(list(missing)[:5])}\n"
        analysis += f"Extra (in LLM, not in ideal): {', '.join(list(extra)[:5])}\n"

        self.chart_display.delete("1.0", tk.END)
        self.chart_display.insert(tk.END, analysis)

    # ==================== OPTIMIZATION PIPELINE ====================
    def launch_optimization_pipeline(self):
        if not self.test_history:
            messagebox.showwarning("No Tests", "Run 'Estimate & Explain' on at least one property first")
            return

        opt_window = tk.Toplevel(self.root)
        opt_window.title("LLM Training Optimization - Interactive Pipeline")
        opt_window.geometry("1600x950")
        opt_window.configure(bg=self.bg_color)

        # ---- HEADER ----
        header_frame = ttk.Frame(opt_window, style="Card.TFrame")
        header_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(header_frame, text="Interactive LLM Training Pipeline + Judge Evaluation", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header_frame, text=f"v{self.prompt_version} | Tests: {len(self.test_history)} | Iterate -> Test -> Judge -> Improve",
                  font=('Helvetica', 9), foreground="#9ca3af").pack(anchor="w")

        # ---- CONTROL BUTTONS ----
        ctrl_frame = ttk.Frame(opt_window, style="Card.TFrame")
        ctrl_frame.pack(fill="x", padx=15, pady=5)

        def add_new_test():
            self.notebook.select(0)
            opt_window.destroy()
            messagebox.showinfo("Add Test",
                "Select a property\n"
                "Click 'Estimate & Explain'\n"
                "Test is auto-saved\n\n"
                "Done? Reopen Optimization Pipeline to see new results!")

        def track_progress():
            if len(self.test_history) < 1:
                messagebox.showinfo("No data", "Run at least 1 test to track progress")
                return

            prog_window = tk.Toplevel(self.root)
            prog_window.title("Progress Tracking - Metrics Over Time")
            prog_window.geometry("1100x700")
            prog_window.configure(bg=self.bg_color)

            ttk.Label(prog_window, text="Progress Tracking Across Versions", style="Title.TLabel").pack(anchor="w", padx=15, pady=10)

            left_pane = ttk.Frame(prog_window)
            left_pane.pack(side="left", fill="both", expand=True, padx=15, pady=10)

            right_pane = ttk.Frame(prog_window)
            right_pane.pack(side="right", fill="both", expand=True, padx=15, pady=10)

            ttk.Label(left_pane, text="Test History (with Judge Scores)", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
            metrics_text = scrolledtext.ScrolledText(left_pane, height=30, width=60,
                                                      bg="#1e2530", fg="#f3f4f6", relief="flat", font=('Courier', 8))
            metrics_text.pack(fill="both", expand=True)

            header_line = f"{'#':<3} {'v':<2} {'Price%':<10} {'BLEU':<8} {'Cons%':<10} {'ROUGE':<8} {'Judge':<8}\n"
            metrics_text.insert(tk.END, header_line)
            metrics_text.insert(tk.END, "-"*56 + "\n")

            for i, test in enumerate(self.test_history, 1):
                judge_str = f"{test['judge_score']:.1f}/10" if test.get('judge_score') is not None else "N/A"
                metrics_text.insert(tk.END,
                    f"{i:<3} {test['prompt_version']:<2} "
                    f"{test['price_error_llm']:>8.1f}% "
                    f"{test['bleu']:>7.1f}% "
                    f"{test['consistency']:>9.1f}% "
                    f"{test['rouge']:>7.1f}% "
                    f"{judge_str:>7}\n"
                )

            ttk.Label(right_pane, text="Summary Statistics", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
            summary_text = scrolledtext.ScrolledText(right_pane, height=30, width=40,
                                                      bg="#1e2530", fg="#f3f4f6", relief="flat", font=('Courier', 9))
            summary_text.pack(fill="both", expand=True)

            avg_price = sum(t['price_error_llm'] for t in self.test_history) / len(self.test_history)
            avg_bleu = sum(t['bleu'] for t in self.test_history) / len(self.test_history)
            avg_cons = sum(t['consistency'] for t in self.test_history) / len(self.test_history)
            avg_rouge = sum(t['rouge'] for t in self.test_history) / len(self.test_history)

            judged = [t for t in self.test_history if t.get('judge_score') is not None]
            avg_judge = sum(t['judge_score'] for t in judged) / len(judged) if judged else None

            latest = self.test_history[-1]
            first = self.test_history[0]

            summary_text.insert(tk.END, f"CURRENT STATE (v{self.prompt_version})\n")
            summary_text.insert(tk.END, "="*35 + "\n\n")
            summary_text.insert(tk.END, f"Total Tests: {len(self.test_history)}\n")
            summary_text.insert(tk.END, f"Judged Tests: {len(judged)}\n\n")

            summary_text.insert(tk.END, "LATEST TEST METRICS:\n")
            summary_text.insert(tk.END, "-"*35 + "\n")
            summary_text.insert(tk.END, f"Price Error:   {latest['price_error_llm']:.1f}%\n")
            summary_text.insert(tk.END, f"BLEU:          {latest['bleu']:.1f}%\n")
            summary_text.insert(tk.END, f"Consistency:   {latest['consistency']:.1f}%\n")
            summary_text.insert(tk.END, f"ROUGE:         {latest['rouge']:.1f}%\n")
            judge_disp = f"{latest['judge_score']:.1f}/10" if latest.get('judge_score') is not None else "Not judged"
            summary_text.insert(tk.END, f"Judge Score:   {judge_disp}\n\n")

            summary_text.insert(tk.END, "AGGREGATE:\n")
            summary_text.insert(tk.END, "-"*35 + "\n")
            summary_text.insert(tk.END, f"Avg Price Err: {avg_price:.1f}%\n")
            summary_text.insert(tk.END, f"Avg BLEU:      {avg_bleu:.1f}%\n")
            summary_text.insert(tk.END, f"Avg Consist:   {avg_cons:.1f}%\n")
            summary_text.insert(tk.END, f"Avg ROUGE:     {avg_rouge:.1f}%\n")
            if avg_judge is not None:
                summary_text.insert(tk.END, f"Avg Judge:     {avg_judge:.1f}/10\n")

            summary_text.insert(tk.END, "\nPROGRESS (first vs latest):\n")
            summary_text.insert(tk.END, "-"*35 + "\n")
            for label, key in [("Price", 'price_error_llm'), ("BLEU", 'bleu'),
                                ("Consist", 'consistency'), ("ROUGE", 'rouge')]:
                delta = latest[key] - first[key]
                direction = "+" if delta > 0 else ""
                summary_text.insert(tk.END, f"{label}: {direction}{delta:.1f}%\n")

        def launch_judge_analysis():
            """Open LLM-as-a-Judge panel."""
            self._open_judge_panel(opt_window)

        def apply_v2_prompt():
            """Switch to v2 prompt using backend.build_prompt_v2() with SHAP dynamic weights."""
            if not hasattr(self, 'houses') or self.combo_houses.current() < 0:
                messagebox.showwarning("No House", "Select a property in Tab 1 first.")
                return

            house = self.houses[self.combo_houses.current()]
            features = house["features"]

            pred_price, _, contributions = self.backend.get_local_contributions(features)
            v2_prompt = self.backend.build_prompt_v2(features, pred_price, contributions)

            self.current_prompt = v2_prompt
            self.prompt_version = 2
            self.lbl_prompt_version.config(text="v2 (SHAP-Grounded)", foreground="#34d399")

            self.txt_prompt_display.config(state="normal")
            self.txt_prompt_display.delete("1.0", tk.END)
            self.txt_prompt_display.insert(tk.END, v2_prompt)
            self.txt_prompt_display.config(state="disabled")

            messagebox.showinfo("Applied", "Prompt v2 (SHAP-Grounded) activated.\nRun 'Estimate & Explain' to test.")
            opt_window.destroy()

        ttk.Button(ctrl_frame, text="+ Add New Test", command=add_new_test).pack(side="left", padx=5)
        ttk.Button(ctrl_frame, text="Judge Analysis", command=launch_judge_analysis).pack(side="left", padx=5)
        ttk.Button(ctrl_frame, text="Apply v2 Prompt", command=apply_v2_prompt).pack(side="left", padx=5)
        ttk.Button(ctrl_frame, text="Track Progress", command=track_progress).pack(side="left", padx=5)

        # ---- FOUR-PANE LAYOUT ----
        main_frame = ttk.Frame(opt_window)
        main_frame.pack(fill="both", expand=True, padx=15, pady=5)

        # Pane 1: ML vs LLM tests
        left_frame = ttk.Frame(main_frame, style="Card.TFrame")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))

        ttk.Label(left_frame, text="ML vs LLM Tests", style="Header.TLabel").pack(anchor="w", pady=(5, 3))
        left_text = scrolledtext.ScrolledText(left_frame, height=35, width=42,
                                               bg="#1e2530", fg="#f3f4f6", relief="flat", font=('Courier', 7))
        left_text.pack(fill="both", expand=True)

        # Pane 2: Issues
        center_frame = ttk.Frame(main_frame, style="Card.TFrame")
        center_frame.pack(side="left", fill="both", expand=True, padx=4)

        ttk.Label(center_frame, text="Issues & Errors", style="Header.TLabel").pack(anchor="w", pady=(5, 3))
        center_text = scrolledtext.ScrolledText(center_frame, height=35, width=38,
                                                 bg="#1e2530", fg="#f3f4f6", relief="flat", font=('Courier', 7))
        center_text.pack(fill="both", expand=True)

        # Pane 3: Improvements
        right_frame = ttk.Frame(main_frame, style="Card.TFrame")
        right_frame.pack(side="left", fill="both", expand=True, padx=4)

        ttk.Label(right_frame, text="Prompt v2 Improvements", style="Header.TLabel").pack(anchor="w", pady=(5, 3))
        right_text = scrolledtext.ScrolledText(right_frame, height=35, width=42,
                                                bg="#1e2530", fg="#d1d5db", relief="flat", font=('Courier', 7))
        right_text.pack(fill="both", expand=True)

        # Pane 4: Judge Summary (NEW)
        judge_frame = ttk.Frame(main_frame, style="Card.TFrame")
        judge_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))

        ttk.Label(judge_frame, text="Judge Scores (LLM-as-a-Judge)", style="Header.TLabel").pack(anchor="w", pady=(5, 3))
        judge_text = scrolledtext.ScrolledText(judge_frame, height=35, width=42,
                                                bg="#1e2530", fg="#a78bfa", relief="flat", font=('Courier', 7))
        judge_text.pack(fill="both", expand=True)

        opt_window.right_text_ref = right_text

        # ---- POPULATE PANES ----
        try:
            # Coverage check before populating
            if hasattr(self, 'houses') and self.houses:
                thin = self.backend.coverage_check(self.houses[:20])
                if thin:
                    center_text.insert(tk.END, "COVERAGE WARNINGS:\n" + "="*38 + "\n")
                    center_text.insert(tk.END, f"{len(thin)} houses have thin keyword\n")
                    center_text.insert(tk.END, "coverage (< 3 keywords for top factor).\n")
                    center_text.insert(tk.END, "This may deflate consistency scores.\n\n")
                    for t in thin[:5]:
                        center_text.insert(tk.END,
                            f"  House {t['house_id']}: {t['factor_label']}"
                            f" ({t['keyword_count']} keywords)\n")
                    center_text.insert(tk.END, "\n")

            price_errors_ml, price_errors_llm, bleu_scores, rouge_scores = [], [], [], []
            consistency_scores, error_differences = [], []

            # Pane 1 + 2
            left_text.insert(tk.END, "DETAILED ML vs LLM:\n" + "="*42 + "\n\n")

            for i, test in enumerate(self.test_history, 1):
                ml_error = abs(test['ml_price'] - test['actual_price']) / test['actual_price'] * 100
                llm_error = test['price_error_llm']
                error_diff = llm_error - ml_error

                bleu_scores.append(test['bleu'])
                rouge_scores.append(test['rouge'])
                consistency_scores.append(test['consistency'])
                price_errors_ml.append(ml_error)
                price_errors_llm.append(llm_error)
                error_differences.append(error_diff)

                judge_str = f"{test['judge_score']:.1f}/10" if test.get('judge_score') is not None else "not judged"

                left_text.insert(tk.END, f"Test {i}: House {test['house_id']}\n")
                left_text.insert(tk.END, f"  Actual:  ${test['actual_price']:,.0f}\n")
                left_text.insert(tk.END, f"  ML:      ${test['ml_price']:,.0f} ({ml_error:+.1f}%)\n")
                left_text.insert(tk.END, f"  LLM:     ${test['llm_price']:,.0f} ({llm_error:+.1f}%)\n")
                left_text.insert(tk.END, f"  Judge:   {judge_str}\n")
                halluc = test.get('hallucinations', [])
                omis = test.get('omissions', [])
                hint = test.get('shap_hint_strength', '--')
                left_text.insert(tk.END, f"  Halluc:  {len(halluc)} | Omis: {len(omis)} | Hint: {hint}\n")
                left_text.insert(tk.END, f"  LLM vs ML: {error_diff:+.1f}%\n\n")

                if error_diff > 20:
                    center_text.insert(tk.END, f"X Test {i}: LLM +{error_diff:.1f}% error vs ML\n\n")
                if test['bleu'] < 15:
                    center_text.insert(tk.END, f"X Test {i}: Wording mismatch (BLEU {test['bleu']:.1f}%)\n\n")
                if test['consistency'] < 70:
                    center_text.insert(tk.END, f"X Test {i}: Missing factors (Cons {test['consistency']:.1f}%)\n\n")
                if test.get('judge_score') is not None and test['judge_score'] < 5:
                    center_text.insert(tk.END, f"X Test {i}: Low judge score ({test['judge_score']:.1f}/10)\n")
                    if test.get('judge_reason'):
                        center_text.insert(tk.END, f"   Reason: {test['judge_reason']}\n\n")

            avg_ml = sum(price_errors_ml) / len(price_errors_ml) if price_errors_ml else 0
            avg_llm = sum(price_errors_llm) / len(price_errors_llm) if price_errors_llm else 0
            avg_diff = sum(error_differences) / len(error_differences) if error_differences else 0
            avg_cons = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
            avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
            avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0

            center_text.insert(tk.END, "\n" + "="*35 + "\nCOMPARISON SUMMARY:\n" + "="*35 + "\n\n")
            center_text.insert(tk.END, f"ML avg error:    {avg_ml:.1f}%\n")
            center_text.insert(tk.END, f"LLM avg error:   {avg_llm:.1f}%\n")
            center_text.insert(tk.END, f"LLM worse by:    {avg_diff:+.1f}%\n\n")
            center_text.insert(tk.END, f"BLEU:            {avg_bleu:.1f}%\n")
            center_text.insert(tk.END, f"Consistency:     {avg_cons:.1f}%\n")
            center_text.insert(tk.END, f"ROUGE:           {avg_rouge:.1f}%\n")

            judged_tests = [t for t in self.test_history if t.get('judge_score') is not None]
            if judged_tests:
                avg_j = sum(t['judge_score'] for t in judged_tests) / len(judged_tests)
                center_text.insert(tk.END, f"Avg Judge:       {avg_j:.1f}/10 ({len(judged_tests)} judged)\n")

            # Pane 3: Improvements
            improvements_text = []
            right_text.insert(tk.END, "v2 PROMPT IMPROVEMENTS\n" + "="*40 + "\n\n")

            if avg_diff > 15:
                right_text.insert(tk.END, "1. PRICE ERROR +{:.1f}%\n   Use ML baseline + small adjustments\n\n".format(avg_diff))
                improvements_text.append(("price", "exact_formula"))
            if avg_cons < 75:
                right_text.insert(tk.END, "2. MISSING FACTORS\n   Require explicit mention of all 8 features\n\n")
                improvements_text.append(("factors", "all_8"))
            if avg_bleu < 20:
                right_text.insert(tk.END, "3. WORDING MATCH\n   Use standard terms: median income, house age\n\n")
                improvements_text.append(("terms", "standard"))
            if avg_rouge < 35:
                right_text.insert(tk.END, "4. ANALYSIS DEPTH\n   Show step-by-step calculations\n\n")
                improvements_text.append(("depth", "detailed"))

            if judged_tests and avg_j < 6:
                right_text.insert(tk.END, "5. JUDGE FEEDBACK\n")
                right_text.insert(tk.END, f"   Avg Judge Score: {avg_j:.1f}/10 - below threshold\n")
                for t in judged_tests[-3:]:
                    if t.get('judge_reason'):
                        right_text.insert(tk.END, f"   '{t['judge_reason']}'\n")
                right_text.insert(tk.END, "\n")
                improvements_text.append(("judge", "low_score"))

            opt_window.improvements_to_apply = improvements_text

            right_text.insert(tk.END, "="*40 + "\n")
            right_text.insert(tk.END, "Click 'Apply v2 Prompt' to activate\n\n")
            right_text.insert(tk.END, self.generate_iterative_prompt(improvements_text, avg_bleu, avg_cons))

            # Pane 4: Judge Summary (NEW)
            self._populate_judge_pane(judge_text, judged_tests)

        except Exception as e:
            right_text.insert(tk.END, f"\nError: {str(e)}")

    # ==================== JUDGE PANEL ====================
    def _populate_judge_pane(self, judge_text, judged_tests):
        """Populate the Judge pane inside the pipeline with existing scores."""
        judge_text.insert(tk.END, "LLM-AS-A-JUDGE OVERVIEW\n" + "="*40 + "\n\n")

        if not judged_tests:
            judge_text.insert(tk.END, "No judge scores yet.\n\n")
            judge_text.insert(tk.END, "Click 'Judge Analysis' button\n")
            judge_text.insert(tk.END, "to score existing explanations\n")
            judge_text.insert(tk.END, "or run judge on new tests.\n\n")
            judge_text.insert(tk.END, "WHAT JUDGE EVALUATES:\n")
            judge_text.insert(tk.END, "-"*40 + "\n")
            judge_text.insert(tk.END, "A second LLM reads the explanation\n")
            judge_text.insert(tk.END, "and compares it to SHAP ground truth.\n\n")
            judge_text.insert(tk.END, "Score 9-10: top factor identified\n")
            judge_text.insert(tk.END, "Score 5-6:  real factor, missed top\n")
            judge_text.insert(tk.END, "Score 1-3:  contradicts SHAP truth\n\n")
            judge_text.insert(tk.END, "Also runs:\n")
            judge_text.insert(tk.END, "- Pairwise A vs B comparison\n")
            judge_text.insert(tk.END, "- Position bias test (A/B swap)\n")
            return

        scores = [t['judge_score'] for t in judged_tests]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        judge_text.insert(tk.END, f"Tests judged: {len(judged_tests)}\n")
        judge_text.insert(tk.END, f"Avg score:  {avg_score:.2f}/10\n")
        judge_text.insert(tk.END, f"Min/Max:    {min_score:.1f} / {max_score:.1f}\n\n")

        # Distribution
        bins = {"1-3 (bad)": 0, "4-6 (ok)": 0, "7-8 (good)": 0, "9-10 (exc)": 0}
        for s in scores:
            if s <= 3: bins["1-3 (bad)"] += 1
            elif s <= 6: bins["4-6 (ok)"] += 1
            elif s <= 8: bins["7-8 (good)"] += 1
            else: bins["9-10 (exc)"] += 1

        judge_text.insert(tk.END, "SCORE DISTRIBUTION:\n" + "-"*40 + "\n")
        for bucket, count in bins.items():
            bar = "#" * count
            judge_text.insert(tk.END, f"  {bucket}: {bar} ({count})\n")

        judge_text.insert(tk.END, "\nPER-TEST SCORES:\n" + "-"*40 + "\n")
        for i, t in enumerate(judged_tests, 1):
            stars = int(t['judge_score'])
            judge_text.insert(tk.END, f"Test {i} (House {t['house_id']}): {t['judge_score']:.1f}/10\n")
            if t.get('judge_reason'):
                judge_text.insert(tk.END, f"  Reason: {t['judge_reason']}\n")
            if t.get('top_real_factors'):
                judge_text.insert(tk.END, f"  SHAP top: {', '.join(t['top_real_factors'][:2])}\n")
            judge_text.insert(tk.END, "\n")

        # Correlation hint
        if len(judged_tests) >= 2:
            judge_text.insert(tk.END, "="*40 + "\n")
            judge_text.insert(tk.END, "CORRELATION HINT:\n")
            keyword_scores = [t['consistency'] for t in judged_tests]
            judge_scores_list = [t['judge_score'] for t in judged_tests]
            if len(keyword_scores) >= 2:
                try:
                    corr = np.corrcoef(keyword_scores, judge_scores_list)[0, 1]
                    judge_text.insert(tk.END, f"Keyword vs Judge corr: {corr:.3f}\n")
                    judge_text.insert(tk.END, "(1.0 = perfect alignment)\n")
                except Exception:
                    pass

    def _open_judge_panel(self, parent_window=None):
        """Full LLM-as-a-Judge analysis window: absolute scores + pairwise + position bias."""
        if not self.test_history:
            messagebox.showwarning("No Data", "Run at least one 'Estimate & Explain' test first.")
            return

        jwin = tk.Toplevel(parent_window or self.root)
        jwin.title("LLM-as-a-Judge - Evaluare Explicatii")
        jwin.geometry("1200x850")
        jwin.configure(bg=self.bg_color)

        # Header
        hdr = ttk.Frame(jwin, style="Card.TFrame")
        hdr.pack(fill="x", padx=15, pady=8)
        ttk.Label(hdr, text="LLM-as-a-Judge — Evaluare Factuala a Explicatiilor", style="Title.TLabel").pack(anchor="w")
        ttk.Label(hdr, text=(
            "Un al doilea LLM judeca fiecare explicatie comparand-o cu adevarul SHAP. "
            "Score 1-10 absolut + comparatie pereche A vs B + test bias de pozitie."
        ), font=('Helvetica', 9), foreground="#9ca3af").pack(anchor="w")

        # Config row
        cfg_frame = ttk.Frame(jwin, style="Card.TFrame")
        cfg_frame.pack(fill="x", padx=15, pady=5)

        ttk.Label(cfg_frame, text="Mode:", style="CardText.TLabel").pack(side="left", padx=(0, 5))
        judge_mode_var = tk.StringVar(value="absolute")
        ttk.Radiobutton(cfg_frame, text="Absolute (1-10 per test)", variable=judge_mode_var, value="absolute").pack(side="left", padx=5)
        ttk.Radiobutton(cfg_frame, text="Pairwise (A vs B)", variable=judge_mode_var, value="pairwise").pack(side="left", padx=5)
        ttk.Radiobutton(cfg_frame, text="Position Bias Test", variable=judge_mode_var, value="bias").pack(side="left", padx=5)

        ttk.Label(cfg_frame, text="  Force simulation:", style="CardText.TLabel").pack(side="left", padx=(15, 5))
        force_sim_judge_var = tk.BooleanVar(value=self.force_sim_var.get())
        ttk.Checkbutton(cfg_frame, text="(no LLM needed)", variable=force_sim_judge_var).pack(side="left")

        # Test selector for pairwise
        ttk.Label(cfg_frame, text="   Test A:", style="CardText.TLabel").pack(side="left", padx=(15, 5))
        test_options = [f"Test {i+1} - House {t['house_id']}" for i, t in enumerate(self.test_history)]
        combo_a = ttk.Combobox(cfg_frame, values=test_options, width=22, state="readonly")
        combo_a.pack(side="left", padx=2)
        if test_options:
            combo_a.current(0)

        ttk.Label(cfg_frame, text="vs B:", style="CardText.TLabel").pack(side="left", padx=(5, 2))
        combo_b = ttk.Combobox(cfg_frame, values=test_options, width=22, state="readonly")
        combo_b.pack(side="left", padx=2)
        if len(test_options) > 1:
            combo_b.current(1)

        # Main content — two panes
        content = ttk.Frame(jwin)
        content.pack(fill="both", expand=True, padx=15, pady=5)

        left_pane = ttk.Frame(content, style="Card.TFrame")
        left_pane.pack(side="left", fill="both", expand=True, padx=(0, 8))

        ttk.Label(left_pane, text="Rezultate Judge", style="Header.TLabel").pack(anchor="w", pady=(5, 3))
        result_text = scrolledtext.ScrolledText(left_pane, height=30, width=65,
                                                 bg="#1e2530", fg="#a78bfa", relief="flat", font=('Courier', 8))
        result_text.pack(fill="both", expand=True)

        right_pane = ttk.Frame(content, style="Card.TFrame", width=340)
        right_pane.pack(side="right", fill="both", expand=False)
        right_pane.pack_propagate(False)

        ttk.Label(right_pane, text="Score Summary", style="Header.TLabel").pack(anchor="w", pady=(5, 3))
        summary_text = scrolledtext.ScrolledText(right_pane, height=15, width=40,
                                                  bg="#1e2530", fg="#f3f4f6", relief="flat", font=('Courier', 8))
        summary_text.pack(fill="x", padx=5, pady=5)

        # Score chart placeholder
        fig_j, ax_j = plt.subplots(figsize=(3.5, 2.8))
        fig_j.patch.set_facecolor('#273142')
        ax_j.set_facecolor('#1e2530')
        ax_j.set_title("Judge Scores per Test", color=self.fg_color, fontsize=8)
        canvas_j = FigureCanvasTkAgg(fig_j, master=right_pane)
        canvas_j.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Status bar
        status_var = tk.StringVar(value="Ready — click 'Run Judge' to start.")
        status_lbl = ttk.Label(jwin, textvariable=status_var, font=('Helvetica', 9), foreground="#34d399")
        status_lbl.pack(anchor="w", padx=15, pady=3)

        # Progress bar
        prog_bar = ttk.Progressbar(jwin, mode='determinate', maximum=100)
        prog_bar.pack(fill="x", padx=15, pady=(0, 5))

        # ---- RUN BUTTON ----
        def run_judge():
            mode = judge_mode_var.get()
            force_sim = force_sim_judge_var.get()

            result_text.delete("1.0", tk.END)
            summary_text.delete("1.0", tk.END)

            btn_run.config(state="disabled")
            status_var.set("Running judge evaluation...")
            jwin.update()

            def do_run():
                try:
                    if mode == "absolute":
                        _run_absolute(force_sim)
                    elif mode == "pairwise":
                        _run_pairwise(force_sim)
                    elif mode == "bias":
                        _run_bias(force_sim)
                except Exception as ex:
                    result_text.insert(tk.END, f"\nEroare: {ex}")
                finally:
                    btn_run.config(state="normal")
                    status_var.set("Done.")

            # Run in thread to keep GUI responsive
            t = threading.Thread(target=do_run, daemon=True)
            t.start()

        def _run_absolute(force_sim):
            """Score each test individually (1-10)."""
            tests = self.test_history
            scores_collected = []
            n = len(tests)

            result_text.insert(tk.END, "ABSOLUTE JUDGE SCORES (1-10 per explicatie)\n")
            result_text.insert(tk.END, "="*60 + "\n\n")
            result_text.insert(tk.END, f"Evaluez {n} teste...\n\n")

            for i, test in enumerate(tests):
                status_var.set(f"Judging test {i+1}/{n}: House {test['house_id']}...")
                prog_bar['value'] = (i / n) * 100
                jwin.update()

                explanation = test.get('explanation_text', '')
                if not explanation:
                    result_text.insert(tk.END, f"Test {i+1} (House {test['house_id']}): NO EXPLANATION — skipped\n\n")
                    continue

                top_factors = test.get('top_real_factors', [])

                judge_result, was_sim = self.backend.llm_judge_score(
                    explanation, top_factors, force_simulation=force_sim
                )
                score = judge_result['score']
                reason = judge_result['reason']
                sim_label = " [SIMULATED]" if was_sim else " [LLM]"

                # Update test record
                test['judge_score'] = score
                test['judge_reason'] = reason
                scores_collected.append(score)

                result_text.insert(tk.END, f"Test {i+1} — House {test['house_id']}{sim_label}\n")
                result_text.insert(tk.END, f"  SHAP Top Factors: {', '.join(top_factors[:3])}\n")
                result_text.insert(tk.END, f"  Explanation: {explanation[:150]}...\n")
                result_text.insert(tk.END, f"  JUDGE SCORE: {score:.1f}/10\n")
                result_text.insert(tk.END, f"  Reason: {reason}\n\n")

                # Color feedback
                if score >= 8:
                    result_text.insert(tk.END, "  [EXCELLENT - factually consistent]\n\n")
                elif score >= 5:
                    result_text.insert(tk.END, "  [OK - partial alignment]\n\n")
                else:
                    result_text.insert(tk.END, "  [POOR - contradicts SHAP ground truth]\n\n")

            prog_bar['value'] = 100
            jwin.update()

            # Summary
            if scores_collected:
                avg = sum(scores_collected) / len(scores_collected)
                summary_text.insert(tk.END, f"ABSOLUTE JUDGE SUMMARY\n{'='*35}\n\n")
                summary_text.insert(tk.END, f"Tests evaluated: {len(scores_collected)}\n")
                summary_text.insert(tk.END, f"Average score:   {avg:.2f}/10\n")
                summary_text.insert(tk.END, f"Min score:       {min(scores_collected):.1f}\n")
                summary_text.insert(tk.END, f"Max score:       {max(scores_collected):.1f}\n\n")

                # Bootstrap CI
                ci = self.backend.bootstrap_ci(
                    [{'consistency_score': s} for s in scores_collected],
                    metric_key='consistency_score'
                )
                summary_text.insert(tk.END, f"Bootstrap 95% CI:\n")
                summary_text.insert(tk.END, f"  [{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]\n\n")

                # Verdict
                if avg >= 7:
                    verdict = "GOOD — LLM aligns with SHAP"
                elif avg >= 4.5:
                    verdict = "PARTIAL — some misalignment"
                else:
                    verdict = "POOR — LLM contradicts SHAP"
                summary_text.insert(tk.END, f"Verdict: {verdict}\n")

                # Update chart
                _update_chart(scores_collected)

                result_text.insert(tk.END, "="*60 + "\n")
                result_text.insert(tk.END, f"MEDIE GENERALA: {avg:.2f}/10 | {verdict}\n")

        def _run_pairwise(force_sim):
            """Judge A vs B pairwise for selected tests."""
            idx_a = combo_a.current()
            idx_b = combo_b.current()

            if idx_a < 0 or idx_b < 0:
                result_text.insert(tk.END, "Selecteaza Test A si Test B!\n")
                return

            if idx_a == idx_b:
                result_text.insert(tk.END, "Selecteaza teste DIFERITE pentru A si B!\n")
                return

            test_a = self.test_history[idx_a]
            test_b = self.test_history[idx_b]

            exp_a = test_a.get('explanation_text', '')
            exp_b = test_b.get('explanation_text', '')

            if not exp_a or not exp_b:
                result_text.insert(tk.END, "Unul dintre teste nu are explicatie salvata.\n")
                return

            # Use top factors from test A (same ground truth basis)
            top_factors = test_a.get('top_real_factors') or test_b.get('top_real_factors') or []

            result_text.insert(tk.END, "PAIRWISE JUDGE: A vs B\n" + "="*60 + "\n\n")
            result_text.insert(tk.END, f"Test A: House {test_a['house_id']} (Test #{idx_a+1})\n")
            result_text.insert(tk.END, f"  Explanation A: {exp_a[:200]}...\n\n")
            result_text.insert(tk.END, f"Test B: House {test_b['house_id']} (Test #{idx_b+1})\n")
            result_text.insert(tk.END, f"  Explanation B: {exp_b[:200]}...\n\n")
            result_text.insert(tk.END, f"SHAP Ground Truth: {', '.join(top_factors[:3])}\n\n")

            status_var.set("Running pairwise judge (A vs B)...")
            prog_bar['value'] = 50
            jwin.update()

            if force_sim:
                result_text.insert(tk.END, "[SIMULATION MODE — no real LLM]\n")
                result_text.insert(tk.END, "Pairwise judge necesita LLM real.\n")
                result_text.insert(tk.END, "Dezactiveaza 'Force simulation' pentru a rula.\n")
                prog_bar['value'] = 100
                return

            pairwise_result, failed = self.backend.llm_judge_pairwise(
                exp_a, exp_b, top_factors
            )

            prog_bar['value'] = 100
            jwin.update()

            if failed or pairwise_result is None:
                result_text.insert(tk.END, "PAIRWISE JUDGE UNAVAILABLE\n")
                result_text.insert(tk.END, "Ollama and Groq are not available.\n")
                result_text.insert(tk.END, "Make sure Ollama is running locally (port 11434)\n")
                result_text.insert(tk.END, "or set GROQ_API_KEY in environment.\n")
            else:
                winner = pairwise_result['winner']
                reason = pairwise_result['reason']
                winner_test = test_a if winner == "A" else test_b

                result_text.insert(tk.END, f"WINNER: {winner} (House {winner_test['house_id']})\n")
                result_text.insert(tk.END, f"Reason: {reason}\n\n")

                summary_text.insert(tk.END, "PAIRWISE RESULT\n" + "="*35 + "\n\n")
                summary_text.insert(tk.END, f"Winner: {winner}\n")
                summary_text.insert(tk.END, f"House {winner_test['house_id']} has better\n")
                summary_text.insert(tk.END, f"factual alignment per judge.\n\n")
                summary_text.insert(tk.END, f"Reason:\n{reason}\n")

        def _run_bias(force_sim):
            """Position bias test: run A vs B then swap B vs A and check consistency."""
            idx_a = combo_a.current()
            idx_b = combo_b.current()

            if idx_a < 0 or idx_b < 0 or idx_a == idx_b:
                result_text.insert(tk.END, "Select two DIFFERENT tests!\n")
                return

            test_a = self.test_history[idx_a]
            test_b = self.test_history[idx_b]
            exp_a = test_a.get('explanation_text', '')
            exp_b = test_b.get('explanation_text', '')
            top_factors = test_a.get('top_real_factors') or test_b.get('top_real_factors') or []

            result_text.insert(tk.END, "POSITION BIAS TEST\n" + "="*60 + "\n\n")
            result_text.insert(tk.END, "Method: the judge runs twice — once with A first,\n")
            result_text.insert(tk.END, "once with B first. If the verdict differs => position bias.\n\n")

            if force_sim:
                result_text.insert(tk.END, "[SIMULATION — cannot test real bias without LLM]\n")
                return

            status_var.set("Running position bias test (pass 1/2)...")
            prog_bar['value'] = 25
            jwin.update()

            bias_result = self.backend.test_position_bias(exp_a, exp_b, top_factors)

            prog_bar['value'] = 100
            jwin.update()

            if not bias_result.get('available'):
                result_text.insert(tk.END, "BIAS TEST UNAVAILABLE\n")
                result_text.insert(tk.END, "Requires LLM (Ollama or Groq) for both runs.\n")
                return

            result_text.insert(tk.END, f"Normal run (A first):   Winner = {bias_result['winner_normal_order']}\n")
            result_text.insert(tk.END, f"  Reason: {bias_result['reason_normal']}\n\n")
            result_text.insert(tk.END, f"Swapped run (B first): Winner mapped = {bias_result['winner_swapped_order_mapped_back']}\n")
            result_text.insert(tk.END, f"  Reason: {bias_result['reason_swapped']}\n\n")

            consistent = bias_result['consistent']
            result_text.insert(tk.END, f"VERDICT: {'CONSISTENT (no position bias)' if consistent else 'INCONSISTENT (position bias detected!)'}\n\n")

            if consistent:
                result_text.insert(tk.END, "The judge gives the same winner regardless of order.\n")
                result_text.insert(tk.END, "=> Robust evaluation, no position favoritism.\n")
            else:
                result_text.insert(tk.END, "The judge changes the verdict when swapping order!\n")
                result_text.insert(tk.END, "=> POSITION BIAS DETECTED — the judge favors the first position.\n")
                result_text.insert(tk.END, "   Conclusion: the pairwise score is NOT fully reliable.\n")

            summary_text.insert(tk.END, "POSITION BIAS TEST\n" + "="*35 + "\n\n")
            summary_text.insert(tk.END, f"Pass 1 winner: {bias_result['winner_normal_order']}\n")
            summary_text.insert(tk.END, f"Pass 2 winner: {bias_result['winner_swapped_order_mapped_back']}\n")
            summary_text.insert(tk.END, f"Consistent: {'YES' if consistent else 'NO'}\n\n")
            if not consistent:
                summary_text.insert(tk.END, "WARNING: position bias\naffects pairwise reliability!\n")

        def _update_chart(scores):
            ax_j.clear()
            ax_j.set_facecolor('#1e2530')
            indices = list(range(1, len(scores) + 1))
            colors = ['#10b981' if s >= 7 else '#f59e0b' if s >= 4.5 else '#f87171' for s in scores]
            ax_j.bar(indices, scores, color=colors, edgecolor='none')
            ax_j.axhline(y=sum(scores)/len(scores), color='#38bdf8', linestyle='--', linewidth=1, label=f'Avg {sum(scores)/len(scores):.1f}')
            ax_j.set_ylim(0, 10.5)
            ax_j.set_xlabel("Test #", color=self.fg_color, fontsize=7)
            ax_j.set_ylabel("Judge Score", color=self.fg_color, fontsize=7)
            ax_j.set_title("Judge Scores per Test", color=self.fg_color, fontsize=8)
            ax_j.tick_params(colors=self.fg_color, labelsize=7)
            ax_j.legend(facecolor='#273142', edgecolor='none', labelcolor=self.fg_color, fontsize=7)
            fig_j.tight_layout()
            canvas_j.draw()

        # Pre-fill chart if already have scores
        existing_scores = [t['judge_score'] for t in self.test_history if t.get('judge_score') is not None]
        if existing_scores:
            _update_chart(existing_scores)

        btn_run = ttk.Button(jwin, text="Run Judge", command=run_judge)
        btn_run.pack(side="bottom", pady=8)

    # ==================== HELPERS ====================
    def generate_iterative_prompt(self, improvements, bleu, consistency):
        prompt = "PROPERTY PRICE ESTIMATION - TRAINING VERSION\n\n[ITERATION IMPROVEMENTS APPLIED]\n"
        for imp in improvements:
            prompt += f"- {imp}\n"
        prompt += (
            "\nANALYZE THIS PROPERTY AND PROVIDE:\n"
            "1. Detailed breakdown of all 8 factors\n"
            "2. Exact price estimate with reasoning\n"
            "3. Factor importance weights (sum=100%)\n\n"
            "MANDATORY RESPONSE STRUCTURE:\n\n"
            "ANALYSIS: [Explain each factor]\n\n"
            "FACTOR IMPORTANCE:\n"
            "[MEDINC: ##] [HOUSEAGE: ##] [AVEROOMS: ##] [AVEBEDRMS: ##] "
            "[POPULATION: ##] [AVEOCCUP: ##] [LATITUDE: ##] [LONGITUDE: ##]\n\n"
            "[PREDICTION: $number]\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "- Mention ALL 8 factors explicitly\n"
            "- Weights MUST sum to exactly 100%\n"
            "- Price between $15,000 and $500,000"
        )
        return prompt

    def show_factor_analysis(self):
        if not hasattr(self, 'current_contributions'):
            messagebox.showwarning("No Data", "Generate explanation first in Tab 1")
            return

        contributions = self.current_contributions
        top_factors = self.current_top_factors
        feature_names = self.backend.feature_names

        abs_contrib = {k: abs(v) for k, v in contributions.items()}
        total = sum(abs_contrib.values()) if sum(abs_contrib.values()) > 0 else 1

        factor_window = tk.Toplevel(self.root)
        factor_window.title("Factor Importance Analysis")
        factor_window.geometry("1000x700")
        factor_window.configure(bg=self.bg_color)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(self.bg_color)

        sorted_factors = sorted(abs_contrib.items(), key=lambda x: x[1], reverse=True)
        factor_labels = [FEATURE_MAP[f]['en'][:15] for f, _ in sorted_factors]
        factor_values = [(v / total) * 100 for _, v in sorted_factors]
        colors = ['#f87171' if f in top_factors else '#60a5fa' for f, _ in sorted_factors]

        ax1.barh(factor_labels, factor_values, color=colors, edgecolor='white', linewidth=1)
        ax1.set_xlabel("Importance (%)", color=self.fg_color)
        ax1.set_title("Factor Importance Ranking", color=self.fg_color, fontsize=12, fontweight='bold')
        ax1.set_facecolor(self.card_bg)
        ax1.tick_params(colors=self.fg_color)

        ax2.axis('off')
        table_data = [["Factor", "Ideal", "LLM", "Top 3"]]
        for factor in feature_names:
            factor_name = FEATURE_MAP[factor]['en'][:15]
            in_ideal = "Y" if FEATURE_MAP[factor]['en'].lower() in self.current_ideal.lower() else "N"
            in_llm = "Y" if FEATURE_MAP[factor]['en'].lower() in self.current_llm.lower() else "N"
            is_top = "*" if factor in top_factors else ""
            table_data.append([factor_name, in_ideal, in_llm, is_top])

        table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.5, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#4f46e5')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor(self.card_bg if i % 2 == 0 else '#1e2530')
                    cell.set_text_props(color=self.fg_color)

        ax2.set_title("Factor Mentions Analysis", color=self.fg_color, fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=factor_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def show_detailed_analysis(self):
        if not hasattr(self, 'current_ideal'):
            messagebox.showwarning("No Data", "Generate explanation first in Tab 1")
            return

        ideal_words = self.current_ideal.lower().split()
        llm_words = self.current_llm.lower().split()

        analysis = "DETAILED N-GRAM ANALYSIS:\n" + "="*50 + "\n\n"
        for n in range(1, 5):
            ideal_ngrams = [' '.join(ideal_words[i:i+n]) for i in range(len(ideal_words)-n+1)]
            llm_ngrams = [' '.join(llm_words[i:i+n]) for i in range(len(llm_words)-n+1)]
            matching = [ng for ng in llm_ngrams if ng in set(ideal_ngrams)]
            coverage = len(matching) / max(len(llm_ngrams), 1) * 100
            analysis += f"{n}-gram Coverage: {len(matching)}/{len(llm_ngrams)} = {coverage:.1f}%\n"
            if matching:
                analysis += f"  Examples: {', '.join(matching[:4])}\n"
            analysis += "\n"

        analysis += "FACTOR MENTION ANALYSIS:\n" + "="*50 + "\n"
        factors_in_ideal = [f for f in FEATURE_MAP.keys() if FEATURE_MAP[f]['en'].lower() in self.current_ideal.lower()]
        factors_in_llm = [f for f in FEATURE_MAP.keys() if FEATURE_MAP[f]['en'].lower() in self.current_llm.lower()]
        matching_factors = set(factors_in_ideal) & set(factors_in_llm)
        missing_factors = set(factors_in_ideal) - set(factors_in_llm)

        analysis += f"Ideal factors: {len(factors_in_ideal)}\nLLM factors: {len(factors_in_llm)}\n"
        analysis += f"Match rate: {len(matching_factors)}/{max(len(factors_in_ideal),1)} = {len(matching_factors)/max(len(factors_in_ideal),1)*100:.1f}%\n"
        if missing_factors:
            analysis += f"Missing: {', '.join([FEATURE_MAP[f]['en'] for f in missing_factors])}\n"

        messagebox.showinfo("Detailed Analysis", analysis)


if __name__ == "__main__":
    plt.close('all')
    root = tk.Tk()
    app = HousingEvaluatorGUI(root)
    root.mainloop()