import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")

from model_utils import HousingEvaluatorBackend, FEATURE_MAP

class HousingEvaluatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Housing Price Evaluator - LLM Factual Consistency & XAI")
        self.root.geometry("1100x850")
        
        self.backend = HousingEvaluatorBackend()
        self.backend.train_model(train_fraction=0.8) # Default training set size 80%
        
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

    def create_widgets(self):
        # Header principal (fara emoji)
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=15, pady=5)
        
        title_label = ttk.Label(header_frame, text="Housing Price Evaluator - LLM Explanation Consistency & XAI", style="Title.TLabel")
        title_label.pack(anchor="w")
        
        desc_label = ttk.Label(header_frame, text="Verification of alignment between qualitative explanations/ponderile LLM and local mathematical importance weights", font=('Helvetica', 9))
        desc_label.pack(anchor="w")
        
        # Tabs control
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text="Individual Prediction & Analysis")
        self.notebook.add(self.tab4, text="BLEU/ROUGE - LLM Text Quality")

        self.setup_tab1()
        self.setup_tab4()

    # ----------------- TAB 1: PREDICTIE SI ANALIZA INDIVIDUALA -----------------
    def setup_tab1(self):
        # Panou control model (antrenare pe fractiuni)
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
        
        # Selectie casa si generare explicatie
        analysis_frame = ttk.Frame(self.tab1, style="Card.TFrame")
        analysis_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        inner_analysis = ttk.Frame(analysis_frame, style="Card.TFrame")
        inner_analysis.pack(fill="both", expand=True, padx=10, pady=8)
        
        ttk.Label(inner_analysis, text="Individual Property Analysis & LLM Alignment", style="Header.TLabel").grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 5))
        
        # Selectie casa
        ttk.Label(inner_analysis, text="Select property:", style="CardText.TLabel").grid(row=1, column=0, sticky="w")
        self.combo_houses = ttk.Combobox(inner_analysis, width=35, state="readonly")
        self.combo_houses.grid(row=1, column=1, sticky="w", padx=5)
        self.combo_houses.bind("<<ComboboxSelected>>", self.on_house_selected)
        
        self.btn_explain = ttk.Button(inner_analysis, text="Estimate & Explain", command=self.explain_individual_house)
        self.btn_explain.grid(row=1, column=2, padx=10)
        
        self.force_sim_var = tk.BooleanVar(value=False)
        chk_sim = ttk.Checkbutton(inner_analysis, text="Force simulation mode (test without Ollama)", variable=self.force_sim_var)
        chk_sim.grid(row=1, column=3, sticky="w", padx=10)
        
        # Grid layout pentru date, comparatie si grafic
        content_frame = ttk.Frame(inner_analysis, style="Card.TFrame")
        content_frame.grid(row=2, column=0, columnspan=5, sticky="nsew", pady=10)
        inner_analysis.rowconfigure(2, weight=1)
        inner_analysis.columnconfigure(4, weight=1)
        
        # Stanga: date casa si preturi
        left_subframe = ttk.Frame(content_frame, style="Card.TFrame", width=320)
        left_subframe.pack(side="left", fill="both", expand=False)
        left_subframe.pack_propagate(False)
        
        self.txt_house_details = tk.Text(left_subframe, height=7, width=40, bg="#1e2530", fg="#f3f4f6", relief="flat", font=('Courier', 9))
        self.txt_house_details.pack(fill="x", pady=5)
        
        self.txt_price_comparison = tk.Text(left_subframe, height=5, width=40, bg="#1e2530", fg="#38bdf8", relief="flat", font=('Courier', 9))
        self.txt_price_comparison.pack(fill="x", pady=5)
        
        # Dreapta: Plot de comparatie ponderi (Matplotlib)
        self.plot_frame = ttk.Frame(content_frame, style="Card.TFrame")
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        self.fig_weights, self.ax_weights = plt.subplots(figsize=(5, 3))
        self.fig_weights.patch.set_facecolor('#273142')
        self.ax_weights.set_facecolor('#1e2530')
        self.canvas_weights = FigureCanvasTkAgg(self.fig_weights, master=self.plot_frame)
        self.canvas_weights.get_tk_widget().pack(fill="both", expand=True)
        
        # Explicatie text LLM
        ttk.Label(inner_analysis, text="Generated explanation and structured LLM block:", style="CardText.TLabel").grid(row=3, column=0, columnspan=5, sticky="w", pady=(5, 2))
        self.txt_explanation = scrolledtext.ScrolledText(inner_analysis, height=4, width=105, bg="#1e2530", fg="#f3f4f6", relief="flat", font=('Helvetica', 9, 'italic'), wrap=tk.WORD)
        self.txt_explanation.grid(row=4, column=0, columnspan=5, sticky="w", pady=5)
        
        # Metrice consistenta
        self.lbl_metrics_tab1 = ttk.Label(inner_analysis, text="Keyword consistency: unevaluated | Weight distribution similarity (L1): unevaluated | LLM estimation error: unevaluated", style="CardText.TLabel", font=('Helvetica', 9, 'bold'))
        self.lbl_metrics_tab1.grid(row=5, column=0, columnspan=5, sticky="w", pady=5)

    def update_train_stats_label(self):
        r2 = self.backend.metrics["r2"]
        mae = self.backend.metrics["mae"]
        size = self.backend.metrics["train_size"]
        self.lbl_train_stats.config(text=f"R-patrat: {r2:.4f} | MAE: ${mae:,.0f} | Number of samples: {size}")

    def train_on_fraction(self):
        try:
            frac = float(self.spin_train_frac.get()) / 100.0
            if frac <= 0.0 or frac > 1.0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Training percentage must be between 10 si 100.")
            return
            
        self.backend.train_model(train_fraction=frac)
        self.update_train_stats_label()
        self.load_houses_list()
        messagebox.showinfo("Training", "Model has been trained pe fractiunea selectata.")
            
    def load_houses_list(self):
        self.houses = self.backend.get_test_houses(n=50)
        house_strings = [f"Casa ID {h['id']} - Actual Price: ${h['actual_price']:,.0f}" for h in self.houses]
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
        
        # Afisare date casa
        self.txt_house_details.delete("1.0", tk.END)
        self.txt_house_details.insert(tk.END, "PROPERTY YESTA:\n")
        self.txt_house_details.insert(tk.END, f"Median Income:    ${features['MedInc']*10:.1f}k/an\n")
        self.txt_house_details.insert(tk.END, f"House Age:    {features['HouseAge']:.0f} ani\n")
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
        
        # Initializare complot gol
        self.ax_weights.clear()
        self.ax_weights.set_title("Comparare ponderi caracteristici (Model vs LLM)", color=self.fg_color, fontsize=9)
        self.fig_weights.tight_layout()
        self.canvas_weights.draw()

    def explain_individual_house(self):
        idx = self.combo_houses.current()
        if idx < 0:
            return
            
        house = self.houses[idx]
        features = house["features"]
        
        self.btn_explain.config(state="disabled")
        self.root.update()
        
        try:
            pred_price, y_base, contributions = self.backend.get_local_contributions(features)
            
            # Obtinere explicatie
            force_sim = self.force_sim_var.get()
            explanation_text, was_simulated = self.backend.generate_explanation(
                features, pred_price, contributions, force_simulation=force_sim
            )
            
            self.txt_explanation.delete("1.0", tk.END)
            self.txt_explanation.insert(tk.END, explanation_text)
            
            # Parsare si evaluare
            parsed_data = self.backend.parse_llm_explanation(explanation_text)
            eval_res = self.backend.evaluate_explanation_consistency(contributions, parsed_data)
            
            # Afisare estimari de pret
            llm_pred = eval_res["llm_prediction"]
            actual_price = house["actual_price"]
            
            self.txt_price_comparison.delete("1.0", tk.END)
            self.txt_price_comparison.insert(tk.END, "PRICE ESTIMATES:\n")
            self.txt_price_comparison.insert(tk.END, f"Actual Price:       ${actual_price:,.0f}\n")
            self.txt_price_comparison.insert(tk.END, f"ML Model Price:   ${pred_price:,.0f}\n")
            
            if llm_pred is not None:
                pe = abs(llm_pred - actual_price) / actual_price * 100.0
                self.txt_price_comparison.insert(tk.END, f"LLM Estimate: ${llm_pred:,.0f}\n")
                self.txt_price_comparison.insert(tk.END, f"LLM Price Error:   {pe:.1f}%\n")
                err_text = f"{pe:.1f}%"
            else:
                self.txt_price_comparison.insert(tk.END, "LLM Estimate: N/A (Error format)\n")
                self.txt_price_comparison.insert(tk.END, "LLM Price Error:   N/A\n")
                err_text = "N/A"
                
            # Afisare metrice in text
            cons_pct = eval_res['factual_consistency'] * 100.0
            dist_pct = eval_res['distribution_consistency']
            self.lbl_metrics_tab1.config(
                text=f"Keyword consistency: {cons_pct:.1f}% | Asemanare ponderi (L1): {dist_pct:.1f}% | Error pret LLM: {err_text}"
            )
            
            # Desenare grafic de comparare ponderi in Tab 1
            self.draw_weights_comparison_plot(eval_res["real_weights"], eval_res["llm_weights"])

            # Update Tab 4 (BLEU/ROUGE Dashboard) cu datele actuale
            self.update_bleu_rouge_dashboard(features, explanation_text, contributions, eval_res)

        except Exception as e:
            messagebox.showerror("Error", f"Could not generate explanation: {str(e)}")
        finally:
            self.btn_explain.config(state="normal")
            
    def draw_weights_comparison_plot(self, real_weights, llm_weights):
        self.ax_weights.clear()
        
        labels = [FEATURE_MAP[name]["ro"][:10] for name in self.backend.feature_names]
        x = np.arange(len(labels))
        width = 0.35
        
        y_real = [real_weights[name] for name in self.backend.feature_names]
        y_llm = [llm_weights[name] for name in self.backend.feature_names]
        
        self.ax_weights.bar(x - width/2, y_real, width, label='Model RF', color='#06b6d4')
        self.ax_weights.bar(x + width/2, y_llm, width, label='Estimare LLM', color='#8b5cf6')
        
        self.ax_weights.set_ylabel('Importanta (%)', color=self.fg_color, fontsize=8)
        self.ax_weights.set_title('Comparare ponderi: Model de decizie vs Declarat LLM', color=self.fg_color, fontsize=9)
        self.ax_weights.set_xticks(x)
        self.ax_weights.set_xticklabels(labels, rotation=35, ha='right', color=self.fg_color, fontsize=7)
        self.ax_weights.tick_params(colors=self.fg_color, labelsize=7)
        self.ax_weights.legend(facecolor='#273142', edgecolor='none', labelcolor=self.fg_color, fontsize=8)
        
        self.fig_weights.tight_layout()
        self.canvas_weights.draw()

# --------- TAB 4: BLEU/ROUGE - CALITATE TEXT LLM ---------
    def setup_tab4(self):
        header_frame = ttk.Frame(self.tab4, style="Card.TFrame")
        header_frame.pack(fill="x", padx=10, pady=5)

        inner_header = ttk.Frame(header_frame, style="Card.TFrame")
        inner_header.pack(fill="both", padx=10, pady=8)

        ttk.Label(inner_header, text="LLM Text Quality Evaluation - BLEU & ROUGE Analysis", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(inner_header, text="Compare ideal vs LLM explanation. Green=ideal matches, Orange=LLM matches. Formulas show exact calculations.", font=('Helvetica', 9), foreground="#9ca3af").pack(anchor="w")

        # Frame pentru continut
        content_frame = ttk.Frame(self.tab4, style="Card.TFrame")
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Top: Chart area
        chart_frame = ttk.Frame(self.tab4, style="Card.TFrame")
        chart_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(chart_frame, text="Word Frequency & Complexity Analysis:", style="Header.TLabel").pack(anchor="w", pady=(5, 5))
        self.chart_display = tk.Text(chart_frame, height=6, width=120, bg="#1e2530", fg="#d1d5db", relief="flat", font=('Courier', 8))
        self.chart_display.pack(fill="both")

        # Left: Explicatii
        left_frame = ttk.Frame(content_frame, style="Card.TFrame")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ttk.Label(left_frame, text="Explicatia Ideala (din factori reali):", style="Header.TLabel").pack(anchor="w", pady=(5, 5))
        self.txt_ideal_explanation = scrolledtext.ScrolledText(left_frame, height=6, width=50, bg="#1e2530", fg="#10b981", relief="flat", font=('Helvetica', 9), wrap=tk.WORD)
        self.txt_ideal_explanation.pack(fill="both", expand=True, pady=(0, 10))

        ttk.Label(left_frame, text="Explicatia LLM (din Tab 1):", style="Header.TLabel").pack(anchor="w", pady=(5, 5))
        self.txt_llm_explanation_tab4 = scrolledtext.ScrolledText(left_frame, height=6, width=50, bg="#1e2530", fg="#f59e0b", relief="flat", font=('Helvetica', 9), wrap=tk.WORD)
        self.txt_llm_explanation_tab4.pack(fill="both", expand=True)

        # Dreapta: Metrici
        right_frame = ttk.Frame(content_frame, style="Card.TFrame", width=280)
        right_frame.pack(side="right", fill="both", expand=False, padx=(10, 0))
        right_frame.pack_propagate(False)

        ttk.Label(right_frame, text="TEXT QUALITY METRICS:", style="Header.TLabel").pack(anchor="w", pady=(5, 10))

        # BLEU Score with progress bar
        bleu_frame = ttk.Frame(right_frame, style="Card.TFrame")
        bleu_frame.pack(fill="x", pady=5)
        ttk.Label(bleu_frame, text="BLEU Score (N-gram):", style="CardText.TLabel", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        self.lbl_bleu_tab4 = ttk.Label(bleu_frame, text="-- %", font=('Helvetica', 20, 'bold'), foreground="#60a5fa")
        self.lbl_bleu_tab4.pack(anchor="w")

        # BLEU Progress bar
        self.prog_bleu = ttk.Progressbar(bleu_frame, length=200, mode='determinate', maximum=100)
        self.prog_bleu.pack(fill="x", pady=3)

        ttk.Label(bleu_frame, text="Phrase matching (2-word overlap)", style="CardText.TLabel", font=('Helvetica', 8)).pack(anchor="w")

        # ROUGE Score with progress bar
        rouge_frame = ttk.Frame(right_frame, style="Card.TFrame")
        rouge_frame.pack(fill="x", pady=5)
        ttk.Label(rouge_frame, text="ROUGE Score (LCS):", style="CardText.TLabel", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        self.lbl_rouge_tab4 = ttk.Label(rouge_frame, text="-- %", font=('Helvetica', 20, 'bold'), foreground="#f87171")
        self.lbl_rouge_tab4.pack(anchor="w")

        # ROUGE Progress bar
        self.prog_rouge = ttk.Progressbar(rouge_frame, length=200, mode='determinate', maximum=100)
        self.prog_rouge.pack(fill="x", pady=3)

        ttk.Label(rouge_frame, text="Semantic overlap (content similarity)", style="CardText.TLabel", font=('Helvetica', 8)).pack(anchor="w")

        # Quality Score section
        quality_frame = ttk.Frame(right_frame, style="Card.TFrame")
        quality_frame.pack(fill="x", pady=5)
        ttk.Label(quality_frame, text="OVERALL QUALITY:", style="Header.TLabel").pack(anchor="w")
        self.lbl_quality_score = ttk.Label(quality_frame, text="-- %", font=('Helvetica', 18, 'bold'), foreground="#8b5cf6")
        self.lbl_quality_score.pack(anchor="w")
        self.prog_quality = ttk.Progressbar(quality_frame, length=200, mode='determinate', maximum=100)
        self.prog_quality.pack(fill="x", pady=3)
        self.lbl_quality_text = ttk.Label(quality_frame, text="", style="CardText.TLabel", font=('Helvetica', 8))
        self.lbl_quality_text.pack(anchor="w")

        # Stats and Interpretation
        interp_frame = ttk.Frame(right_frame, style="Card.TFrame")
        interp_frame.pack(fill="both", expand=True, pady=10)

        stats_label = ttk.Label(interp_frame, text="DETAILED ANALYSIS:", style="Header.TLabel")
        stats_label.pack(anchor="w", pady=(5, 3))

        self.txt_interpretation = scrolledtext.ScrolledText(interp_frame, height=10, width=35, bg="#1e2530", fg="#d1d5db", relief="flat", font=('Helvetica', 7, 'bold'), wrap=tk.WORD)
        self.txt_interpretation.pack(fill="both", expand=True)

        # Add buttons for interaction
        btn_frame = ttk.Frame(right_frame, style="Card.TFrame")
        btn_frame.pack(fill="x", pady=(5, 0))
        ttk.Button(btn_frame, text="Copy", command=self.copy_comparison).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Details", command=self.show_detailed_analysis).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Factors", command=self.show_factor_analysis).pack(side="left", padx=2)

    def update_bleu_rouge_dashboard(self, features, llm_explanation_text, contributions, eval_res):
        """Update Tab4 with detailed BLEU/ROUGE calculation breakdown + interactive highlighting"""
        # Generate ideal explanation
        ideal_explanation = self.generate_ideal_explanation(features, eval_res["top_real_factors"])

        # Calculate BLEU and ROUGE
        bleu = self.calculate_bleu_score(ideal_explanation, llm_explanation_text)
        rouge = self.calculate_rouge_score(ideal_explanation, llm_explanation_text)

        # Extract text portions
        llm_text_only = llm_explanation_text.split('[')[0].strip() if '[' in llm_explanation_text else llm_explanation_text[:400]

        # Configure highlighting tags
        self.txt_ideal_explanation.tag_configure("match", background="#10b981", foreground="white", font=('Helvetica', 9, 'bold'))
        self.txt_llm_explanation_tab4.tag_configure("match", background="#f59e0b", foreground="white", font=('Helvetica', 9, 'bold'))

        # Display ideal explanation
        self.txt_ideal_explanation.delete("1.0", tk.END)
        self.txt_ideal_explanation.insert(tk.END, ideal_explanation)

        # Display LLM explanation
        self.txt_llm_explanation_tab4.delete("1.0", tk.END)
        self.txt_llm_explanation_tab4.insert(tk.END, llm_text_only)

        # Highlight matching 2-grams in both texts
        ideal_words = ideal_explanation.lower().split()
        llm_words = llm_text_only.lower().split()

        # Find all 2-gram matches
        ideal_2grams = set([' '.join(ideal_words[i:i+2]) for i in range(len(ideal_words)-1)])
        llm_2grams = [' '.join(llm_words[i:i+2]) for i in range(len(llm_words)-1)]
        matching_2grams = [ng for ng in llm_2grams if ng in ideal_2grams]

        # Highlight in ideal explanation
        for i in range(len(ideal_words)-1):
            phrase = ' '.join(ideal_words[i:i+2])
            if phrase in ideal_2grams:
                for match_phrase in matching_2grams:
                    if phrase.lower() == match_phrase.lower():
                        idx = ideal_explanation.lower().find(phrase.lower())
                        if idx >= 0:
                            self.txt_ideal_explanation.tag_add("match", f"1.0+{idx}c", f"1.0+{idx+len(phrase)}c")
                            break

        # Highlight in LLM explanation
        for i in range(len(llm_words)-1):
            phrase = ' '.join(llm_words[i:i+2])
            if phrase in ideal_2grams:
                idx = llm_text_only.lower().find(phrase.lower())
                if idx >= 0:
                    self.txt_llm_explanation_tab4.tag_add("match", f"1.0+{idx}c", f"1.0+{idx+len(phrase)}c")

        # Update scores and progress bars
        self.lbl_bleu_tab4.config(text=f"{bleu:.1f}%")
        self.lbl_rouge_tab4.config(text=f"{rouge:.1f}%")
        self.prog_bleu['value'] = bleu
        self.prog_rouge['value'] = rouge

        # Store values for interactive features
        self.current_bleu = bleu
        self.current_rouge = rouge
        self.current_ideal = ideal_explanation
        self.current_llm = llm_text_only
        self.current_contributions = contributions
        self.current_top_factors = eval_res["top_real_factors"]

        # Calculate overall quality score (weighted average)
        # BLEU: 30%, ROUGE: 30%, Factual consistency: 40%
        factual_consistency = eval_res.get("factual_consistency", 0) * 100
        quality_score = (bleu * 0.3 + rouge * 0.3 + factual_consistency * 0.4)

        self.lbl_quality_score.config(text=f"{quality_score:.1f}%")
        self.prog_quality['value'] = quality_score

        # Quality text recommendation
        if quality_score >= 80:
            quality_text = "Excellent: LLM explanation well-aligned with ML factors"
        elif quality_score >= 60:
            quality_text = "Good: Moderate alignment, some deviations"
        elif quality_score >= 40:
            quality_text = "Fair: Significant differences between ideal & LLM"
        else:
            quality_text = "Poor: Major factual/conceptual misalignment"

        self.lbl_quality_text.config(text=quality_text)

        # Get technical calculations
        ideal_words = len(ideal_explanation.lower().split())
        llm_words = len(llm_text_only.lower().split())

        # Calculate LCS for ROUGE details
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

        # Engineering-level interpretation
        interp = "TECHNICAL METRICS:\n"
        interp += "="*32 + "\n\n"

        interp += f"BLEU: {bleu:.2f}%\n"
        interp += f"Formula: matches / total_ngrams\n"
        interp += f"Details:\n"
        interp += f"  Matches: {self.bleu_matches}\n"
        interp += f"  2-grams: {self.bleu_total}\n"
        interp += f"  = {self.bleu_matches}/{self.bleu_total} = {self.bleu_precision:.2f}%\n"
        interp += f"Ref: {ideal_words} words\n"
        interp += f"Gen: {llm_words} words\n"

        interp += f"\nROUGE: {rouge:.2f}%\n"
        interp += f"Formula: LCS / ref_words\n"
        interp += f"Details:\n"
        interp += f"  LCS length: {lcs}\n"
        interp += f"  Ref words: {ideal_words}\n"
        interp += f"  = {lcs}/{ideal_words} = {(lcs/ideal_words*100):.2f}%\n"
        interp += f"Precision: {(lcs/llm_words*100):.2f}%\n"

        interp += f"\nFactors: {', '.join([FEATURE_MAP[f]['en'][:8] for f in eval_res['top_real_factors'][:2]])}"

        self.txt_interpretation.delete("1.0", tk.END)
        self.txt_interpretation.insert(tk.END, interp)

        # Generate complexity and frequency analysis
        self.update_complexity_analysis(ideal_explanation, llm_text_only)

    def generate_ideal_explanation(self, features, top_factors, contributions=None):
        """Generate detailed ideal explanation with actual contribution values"""
        # Build accurate explanation with real data
        ideal = ""

        # Primary factor analysis
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
                ideal += f"Geographic location at {features['Latitude']:.2f}°N, {features['Longitude']:.2f}°W is the primary pricing factor. "
                if 37 <= features['Latitude'] <= 38 and -124 <= features['Longitude'] <= -121:
                    ideal += "Bay Area coastal positioning provides significant premium value. "
                elif features['Latitude'] < 34 or features['Latitude'] > 39:
                    ideal += "Regional location affects competitive market pricing. "

        # Secondary factors
        if len(top_factors) > 1:
            secondary = top_factors[1]
            sec_label = FEATURE_MAP[secondary]['en']
            ideal += f"Secondary impact from {sec_label.lower()}. "

        if len(top_factors) > 2:
            tertiary = top_factors[2]
            tert_label = FEATURE_MAP[tertiary]['en']
            ideal += f"Tertiary influence from {tert_label.lower()}. "

        # Add context
        ideal += "Combined analysis of these factors determines the property's market valuation."

        return ideal

    def calculate_bleu_score(self, reference, generated, n=2):
        """Calculate BLEU score (n-gram match) with detailed breakdown"""
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()

        if len(gen_tokens) == 0:
            return 0.0

        matches = 0
        total = max(len(gen_tokens) - n + 1, 0)

        if total == 0:
            return 0.0

        from collections import Counter
        ref_ngrams = [' '.join(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
        gen_ngrams = [' '.join(gen_tokens[i:i+n]) for i in range(len(gen_tokens) - n + 1)]

        ref_counts = Counter(ref_ngrams)
        gen_counts = Counter(gen_ngrams)

        for ngram in gen_counts:
            matches += min(gen_counts[ngram], ref_counts.get(ngram, 0))

        bleu_score = (matches / total) * 100 if total > 0 else 0.0

        # Store details for display
        self.bleu_matches = matches
        self.bleu_total = total
        self.bleu_precision = bleu_score

        return bleu_score

    def calculate_rouge_score(self, reference, generated):
        """Calculate ROUGE score (LCS-based semantic overlap) with technical breakdown"""
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()

        if len(ref_tokens) == 0 or len(gen_tokens) == 0:
            return 0.0

        # LCS (Longest Common Subsequence) using dynamic programming
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
        """Copy comparison data to clipboard"""
        comparison = f"BLEU: {self.current_bleu:.2f}%\nROUGE: {self.current_rouge:.2f}%\n\n"
        comparison += f"IDEAL:\n{self.current_ideal}\n\n"
        comparison += f"LLM:\n{self.current_llm}"
        self.root.clipboard_clear()
        self.root.clipboard_append(comparison)
        messagebox.showinfo("Copied", "Comparison data copied to clipboard!")

    def update_complexity_analysis(self, ideal_text, llm_text):
        """Calculate and display text complexity metrics and word analysis"""
        ideal_words = ideal_text.lower().split()
        llm_words = llm_text.lower().split()

        # Word frequency analysis
        from collections import Counter
        ideal_freq = Counter(ideal_words)
        llm_freq = Counter(llm_words)

        # Calculate metrics
        ideal_unique = len(ideal_freq)
        llm_unique = len(llm_freq)
        ideal_avg_len = sum(len(w) for w in ideal_words) / len(ideal_words) if ideal_words else 0
        llm_avg_len = sum(len(w) for w in llm_words) / len(llm_words) if llm_words else 0

        # Missing and extra words
        missing = set(ideal_words) - set(llm_words)
        extra = set(llm_words) - set(ideal_words)

        # Create analysis display
        analysis = ""
        analysis += f"{'IDEAL':<40} | {'LLM':<40}\n"
        analysis += f"{'-'*40} | {'-'*40}\n"
        analysis += f"Words: {len(ideal_words):<34} | Words: {len(llm_words):<34}\n"
        analysis += f"Unique: {ideal_unique:<34} | Unique: {llm_unique:<34}\n"
        analysis += f"Avg word len: {ideal_avg_len:<28} | Avg word len: {llm_avg_len:<28}\n"
        analysis += f"Vocabulary match: {len(set(ideal_words) & set(llm_words)):<25} | Missing words: {len(missing):<25}\n"
        analysis += f"\nMissing (in ideal, not in LLM): {', '.join(list(missing)[:5])}\n"
        analysis += f"Extra (in LLM, not in ideal): {', '.join(list(extra)[:5])}\n"

        self.chart_display.delete("1.0", tk.END)
        self.chart_display.insert(tk.END, analysis)

    def show_factor_analysis(self):
        """Show factor importance with matplotlib chart + table"""
        if not hasattr(self, 'current_contributions'):
            messagebox.showwarning("No Data", "Generate explanation first in Tab 1")
            return

        contributions = self.current_contributions
        top_factors = self.current_top_factors
        feature_names = self.backend.feature_names

        # Calculate percentages
        abs_contrib = {k: abs(v) for k, v in contributions.items()}
        total = sum(abs_contrib.values()) if sum(abs_contrib.values()) > 0 else 1

        # Create new window
        factor_window = tk.Toplevel(self.root)
        factor_window.title("Factor Importance Analysis")
        factor_window.geometry("1000x700")
        factor_window.configure(bg=self.bg_color)

        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(self.bg_color)

        # LEFT: Bar chart of factor importance
        sorted_factors = sorted(abs_contrib.items(), key=lambda x: x[1], reverse=True)
        factor_labels = [FEATURE_MAP[f]['en'][:15] for f, _ in sorted_factors]
        factor_values = [(v / total) * 100 for _, v in sorted_factors]

        colors = ['#f87171' if f in top_factors else '#60a5fa' for f, _ in sorted_factors]

        ax1.barh(factor_labels, factor_values, color=colors, edgecolor='white', linewidth=1)
        ax1.set_xlabel("Importance (%)", color=self.fg_color)
        ax1.set_title("Factor Importance Ranking", color=self.fg_color, fontsize=12, fontweight='bold')
        ax1.set_facecolor(self.card_bg)
        ax1.tick_params(colors=self.fg_color)
        for spine in ax1.spines.values():
            spine.set_color(self.fg_color)
            spine.set_alpha(0.3)

        # RIGHT: Table of mentions
        ax2.axis('off')

        table_data = []
        table_data.append(["Factor", "Ideal", "LLM", "Top 3"])

        for factor in feature_names:
            factor_name = FEATURE_MAP[factor]['en'][:15]
            in_ideal = "✓" if FEATURE_MAP[factor]['en'].lower() in self.current_ideal.lower() else "✗"
            in_llm = "✓" if FEATURE_MAP[factor]['en'].lower() in self.current_llm.lower() else "✗"
            is_top = "★" if factor in top_factors else ""

            table_data.append([factor_name, in_ideal, in_llm, is_top])

        table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.5, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style table
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4f46e5')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor(self.card_bg if i % 2 == 0 else '#1e2530')
                    cell.set_text_props(color=self.fg_color)

        ax2.set_title("Factor Mentions Analysis", color=self.fg_color, fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()

        # Embed matplotlib in tkinter
        canvas = FigureCanvasTkAgg(fig, master=factor_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def show_detailed_analysis(self):
        """Show detailed n-gram and factor analysis"""
        ideal_words = self.current_ideal.lower().split()
        llm_words = self.current_llm.lower().split()

        # Calculate all n-gram matches
        analysis = "DETAILED N-GRAM ANALYSIS:\n"
        analysis += "="*50 + "\n\n"

        for n in range(1, 5):
            ideal_ngrams = [' '.join(ideal_words[i:i+n]) for i in range(len(ideal_words)-n+1)]
            llm_ngrams = [' '.join(llm_words[i:i+n]) for i in range(len(llm_words)-n+1)]

            matching = [ng for ng in llm_ngrams if ng in set(ideal_ngrams)]
            coverage = len(matching) / max(len(llm_ngrams), 1) * 100

            analysis += f"{n}-gram Coverage: {len(matching)}/{len(llm_ngrams)} = {coverage:.1f}%\n"
            if matching:
                analysis += f"  Examples: {', '.join(matching[:4])}\n"
            analysis += "\n"

        # Factor mentions
        analysis += "FACTOR MENTION ANALYSIS:\n"
        analysis += "="*50 + "\n"
        factors_in_ideal = [f for f in FEATURE_MAP.keys() if FEATURE_MAP[f]['en'].lower() in self.current_ideal.lower()]
        factors_in_llm = [f for f in FEATURE_MAP.keys() if FEATURE_MAP[f]['en'].lower() in self.current_llm.lower()]

        matching_factors = set(factors_in_ideal) & set(factors_in_llm)
        missing_factors = set(factors_in_ideal) - set(factors_in_llm)

        analysis += f"Ideal factors: {len(factors_in_ideal)} ({', '.join([FEATURE_MAP[f]['en'][:10] for f in factors_in_ideal[:3]])})\n"
        analysis += f"LLM factors: {len(factors_in_llm)} ({', '.join([FEATURE_MAP[f]['en'][:10] for f in factors_in_llm[:3]])})\n"
        analysis += f"Match rate: {len(matching_factors)}/{max(len(factors_in_ideal), 1)} = {len(matching_factors)/max(len(factors_in_ideal), 1)*100:.1f}%\n"
        if missing_factors:
            analysis += f"Missing factors: {', '.join([FEATURE_MAP[f]['en'] for f in missing_factors])}\n"

        messagebox.showinfo("Detailed Analysis", analysis)

if __name__ == "__main__":
    # Eliminam afisarea ferestrei goale la matplotlib in fundal
    plt.close('all')
    root = tk.Tk()
    app = HousingEvaluatorGUI(root)
    root.mainloop()
