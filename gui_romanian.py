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
        self.root.title("Evaluator Consistenta Factuala LLM si XAI")
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
        
        title_label = ttk.Label(header_frame, text="Sistem de Evaluare a Consistentei Explicațiilor LLM si XAI", style="Title.TLabel")
        title_label.pack(anchor="w")
        
        desc_label = ttk.Label(header_frame, text="Verificarea alinierii intre explicatiile calitative/ponderile LLM si importantele matematice locale", font=('Helvetica', 9))
        desc_label.pack(anchor="w")
        
        # Tabs control
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text="Predictie si Analiza Individuala")
        self.notebook.add(self.tab2, text="Harta Geografica XAI California")
        self.notebook.add(self.tab3, text="Evaluare Batch si Statistici")
        self.notebook.add(self.tab4, text="BLEU/ROUGE - Calitate Text LLM")

        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()

    # ----------------- TAB 1: PREDICTIE SI ANALIZA INDIVIDUALA -----------------
    def setup_tab1(self):
        # Panou control model (antrenare pe fractiuni)
        model_frame = ttk.Frame(self.tab1, style="Card.TFrame")
        model_frame.pack(fill="x", padx=10, pady=5)
        
        inner_model = ttk.Frame(model_frame, style="Card.TFrame")
        inner_model.pack(fill="both", padx=10, pady=8)
        
        ttk.Label(inner_model, text="Configurare Model Regresie", style="Header.TLabel").grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 5))
        
        ttk.Label(inner_model, text="Procent date antrenare (%):", style="CardText.TLabel").grid(row=1, column=0, sticky="w")
        self.spin_train_frac = ttk.Spinbox(inner_model, from_=10, to=100, increment=10, width=5)
        self.spin_train_frac.set(80)
        self.spin_train_frac.grid(row=1, column=1, sticky="w", padx=5)
        
        btn_train = ttk.Button(inner_model, text="Antreneaza Model", command=self.train_on_fraction)
        btn_train.grid(row=1, column=2, padx=15)
        
        self.lbl_train_stats = ttk.Label(inner_model, text="", style="CardText.TLabel")
        self.lbl_train_stats.grid(row=1, column=3, columnspan=2, sticky="w")
        self.update_train_stats_label()
        
        # Selectie casa si generare explicatie
        analysis_frame = ttk.Frame(self.tab1, style="Card.TFrame")
        analysis_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        inner_analysis = ttk.Frame(analysis_frame, style="Card.TFrame")
        inner_analysis.pack(fill="both", expand=True, padx=10, pady=8)
        
        ttk.Label(inner_analysis, text="Analiza Proprietate Individuala si Aliniere LLM", style="Header.TLabel").grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 5))
        
        # Selectie casa
        ttk.Label(inner_analysis, text="Selecteaza locuinta:", style="CardText.TLabel").grid(row=1, column=0, sticky="w")
        self.combo_houses = ttk.Combobox(inner_analysis, width=35, state="readonly")
        self.combo_houses.grid(row=1, column=1, sticky="w", padx=5)
        self.combo_houses.bind("<<ComboboxSelected>>", self.on_house_selected)
        
        self.btn_explain = ttk.Button(inner_analysis, text="Estimeaza si Explica", command=self.explain_individual_house)
        self.btn_explain.grid(row=1, column=2, padx=10)
        
        self.force_sim_var = tk.BooleanVar(value=False)
        chk_sim = ttk.Checkbutton(inner_analysis, text="Forțare mod simulat (test fără Ollama)", variable=self.force_sim_var)
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
        ttk.Label(inner_analysis, text="Explicatie generata si bloc structurat LLM:", style="CardText.TLabel").grid(row=3, column=0, columnspan=5, sticky="w", pady=(5, 2))
        self.txt_explanation = scrolledtext.ScrolledText(inner_analysis, height=4, width=105, bg="#1e2530", fg="#f3f4f6", relief="flat", font=('Helvetica', 9, 'italic'), wrap=tk.WORD)
        self.txt_explanation.grid(row=4, column=0, columnspan=5, sticky="w", pady=5)
        
        # Metrice consistenta
        self.lbl_metrics_tab1 = ttk.Label(inner_analysis, text="Consistenta cuvinte cheie: neevaluata | Asemanare distributie ponderi (L1): neevaluata | Eroare estimare LLM: neevaluata", style="CardText.TLabel", font=('Helvetica', 9, 'bold'))
        self.lbl_metrics_tab1.grid(row=5, column=0, columnspan=5, sticky="w", pady=5)

    def update_train_stats_label(self):
        r2 = self.backend.metrics["r2"]
        mae = self.backend.metrics["mae"]
        size = self.backend.metrics["train_size"]
        self.lbl_train_stats.config(text=f"R-patrat: {r2:.4f} | MAE: ${mae:,.0f} | Numar esantioane: {size}")

    def train_on_fraction(self):
        try:
            frac = float(self.spin_train_frac.get()) / 100.0
            if frac <= 0.0 or frac > 1.0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Eroare", "Procentul date de antrenare trebuie sa fie intre 10 si 100.")
            return
            
        self.backend.train_model(train_fraction=frac)
        self.update_train_stats_label()
        self.load_houses_list()
        messagebox.showinfo("Antrenare", "Modelul a fost antrenat pe fractiunea selectata.")
            
    def load_houses_list(self):
        self.houses = self.backend.get_test_houses(n=50)
        house_strings = [f"Casa ID {h['id']} - Pret Real: ${h['actual_price']:,.0f}" for h in self.houses]
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
        self.txt_house_details.insert(tk.END, "DATE PROPRIETATE:\n")
        self.txt_house_details.insert(tk.END, f"Venit Median:    ${features['MedInc']*10:.1f}k/an\n")
        self.txt_house_details.insert(tk.END, f"Varsta Medie:    {features['HouseAge']:.0f} ani\n")
        self.txt_house_details.insert(tk.END, f"Numar Camere:    {features['AveRooms']:.2f}\n")
        self.txt_house_details.insert(tk.END, f"Dormitoare:      {features['AveBedrms']:.2f}\n")
        self.txt_house_details.insert(tk.END, f"Populatie Zona:  {int(features['Population'])}\n")
        self.txt_house_details.insert(tk.END, f"Ocupare Medie:   {features['AveOccup']:.2f}\n")
        self.txt_house_details.insert(tk.END, f"Coordonate:      {features['Latitude']:.2f}N, {features['Longitude']:.2f}W")
        
        self.txt_price_comparison.delete("1.0", tk.END)
        self.txt_price_comparison.insert(tk.END, "ESTIMARI DE PRET:\n")
        self.txt_price_comparison.insert(tk.END, f"Pret Real:       ${house['actual_price']:,.0f}\n")
        self.txt_price_comparison.insert(tk.END, f"Pret Model RF:   ${house['predicted_price']:,.0f}\n")
        self.txt_price_comparison.insert(tk.END, "Pret Estimare LLM: Necalculat\n")
        self.txt_price_comparison.insert(tk.END, "Eroare Pret LLM:  Necalculat")
        
        self.txt_explanation.delete("1.0", tk.END)
        self.lbl_metrics_tab1.config(text="Consistenta cuvinte cheie: neevaluata | Asemanare distributie ponderi (L1): neevaluata | Eroare estimare LLM: neevaluata")
        
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
            self.txt_price_comparison.insert(tk.END, "ESTIMARI DE PRET:\n")
            self.txt_price_comparison.insert(tk.END, f"Pret Real:       ${actual_price:,.0f}\n")
            self.txt_price_comparison.insert(tk.END, f"Pret Model RF:   ${pred_price:,.0f}\n")
            
            if llm_pred is not None:
                pe = abs(llm_pred - actual_price) / actual_price * 100.0
                self.txt_price_comparison.insert(tk.END, f"Pret Estimare LLM: ${llm_pred:,.0f}\n")
                self.txt_price_comparison.insert(tk.END, f"Eroare Pret LLM:   {pe:.1f}%\n")
                err_text = f"{pe:.1f}%"
            else:
                self.txt_price_comparison.insert(tk.END, "Pret Estimare LLM: N/A (Eroare format)\n")
                self.txt_price_comparison.insert(tk.END, "Eroare Pret LLM:   N/A\n")
                err_text = "N/A"
                
            # Afisare metrice in text
            cons_pct = eval_res['factual_consistency'] * 100.0
            dist_pct = eval_res['distribution_consistency']
            self.lbl_metrics_tab1.config(
                text=f"Consistenta cuvinte cheie: {cons_pct:.1f}% | Asemanare ponderi (L1): {dist_pct:.1f}% | Eroare pret LLM: {err_text}"
            )
            
            # Desenare grafic de comparare ponderi in Tab 1
            self.draw_weights_comparison_plot(eval_res["real_weights"], eval_res["llm_weights"])

            # Update Tab 4 (BLEU/ROUGE Dashboard) cu datele actuale
            self.update_bleu_rouge_dashboard(features, explanation_text, contributions, eval_res)

        except Exception as e:
            messagebox.showerror("Eroare", f"Nu s-a putut genera explicatia: {str(e)}")
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

    # ----------------- TAB 2: HARTA GEOGRAFICA XAI -----------------
    def setup_tab2(self):
        # Panou configurare harta
        map_control = ttk.Frame(self.tab2, style="Card.TFrame")
        map_control.pack(fill="x", padx=10, pady=5)
        
        inner_map = ttk.Frame(map_control, style="Card.TFrame")
        inner_map.pack(fill="both", padx=10, pady=8)
        
        ttk.Label(inner_map, text="Configurare Vizualizare Spatiala XAI (Coordonate California)", style="Header.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 5))
        
        ttk.Label(inner_map, text="Coloreaza punctele de pe harta dupa:", style="CardText.TLabel").grid(row=1, column=0, sticky="w")
        
        self.combo_map_color = ttk.Combobox(inner_map, width=30, state="readonly")
        self.combo_map_color['values'] = [
            "Pret Real",
            "Pret Prezis (Random Forest)",
            "Eroare Model (Absoluta)",
            "Factor Dominant (Model ML)",
            "Factor Dominant (LLM Estimare)"
        ]
        self.combo_map_color.current(0)
        self.combo_map_color.grid(row=1, column=1, sticky="w", padx=10)
        self.combo_map_color.bind("<<ComboboxSelected>>", lambda e: self.draw_geographic_map())
        
        btn_draw_map = ttk.Button(inner_map, text="Deseneaza Harta", command=self.draw_geographic_map)
        btn_draw_map.grid(row=1, column=2, padx=15)
        
        # Frame pentru canvas
        self.map_canvas_frame = ttk.Frame(self.tab2, style="Card.TFrame")
        self.map_canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.fig_map, self.ax_map = plt.subplots(figsize=(7, 5))
        self.fig_map.patch.set_facecolor('#1e2530')
        self.ax_map.set_facecolor('#273142')
        
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=self.map_canvas_frame)
        self.canvas_widget = self.canvas_map.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
        
        self.toolbar_map = NavigationToolbar2Tk(self.canvas_map, self.map_canvas_frame)
        self.toolbar_map.update()
        self.toolbar_map.pack(fill="x", side="bottom")

    def draw_geographic_map(self):
        if not self.backend.is_trained:
            messagebox.showwarning("Atentie", "Modelul trebuie sa fie antrenat mai intai.")
            return
            
        self.ax_map.clear()
        
        # Extragem esantioanele din setul de test (folosim primele 100 de case pentru a avea o densitate mai buna)
        test_subset = self.backend.X_test.head(100)
        y_subset = self.backend.y_test.head(100)
        
        lats = test_subset["Latitude"].values
        lons = test_subset["Longitude"].values
        
        color_option = self.combo_map_color.get()
        
        if color_option == "Pret Real":
            prices = y_subset.values / 1000.0 # Convertit in mii dolari pentru lizibilitate
            sc = self.ax_map.scatter(lons, lats, c=prices, cmap="coolwarm", s=35, alpha=0.8, edgecolor="none")
            cbar = self.fig_map.colorbar(sc, ax=self.ax_map, shrink=0.8)
            cbar.set_label("Pret Real (mii $)", color=self.fg_color, fontsize=8)
            cbar.ax.yaxis.set_tick_params(color=self.fg_color, labelcolor=self.fg_color)
            self.ax_map.set_title("Harta Geografica a Preturilor Reale in California", color=self.fg_color, fontsize=10)
            
        elif color_option == "Pret Prezis (Random Forest)":
            preds = self.backend.model.predict(test_subset) / 1000.0
            sc = self.ax_map.scatter(lons, lats, c=preds, cmap="coolwarm", s=35, alpha=0.8, edgecolor="none")
            cbar = self.fig_map.colorbar(sc, ax=self.ax_map, shrink=0.8)
            cbar.set_label("Pret Prezis (mii $)", color=self.fg_color, fontsize=8)
            cbar.ax.yaxis.set_tick_params(color=self.fg_color, labelcolor=self.fg_color)
            self.ax_map.set_title("Harta Preturilor Prezise de Modelul ML", color=self.fg_color, fontsize=10)
            
        elif color_option == "Eroare Model (Absoluta)":
            preds = self.backend.model.predict(test_subset)
            errors = np.abs(preds - y_subset.values) / 1000.0
            sc = self.ax_map.scatter(lons, lats, c=errors, cmap="YlOrRd", s=35, alpha=0.8, edgecolor="none")
            cbar = self.fig_map.colorbar(sc, ax=self.ax_map, shrink=0.8)
            cbar.set_label("Eroare Absoluta (mii $)", color=self.fg_color, fontsize=8)
            cbar.ax.yaxis.set_tick_params(color=self.fg_color, labelcolor=self.fg_color)
            self.ax_map.set_title("Distributia Erorilor Absolute ale Modelului ML", color=self.fg_color, fontsize=10)
            
        elif color_option == "Factor Dominant (Model ML)":
            # Pentru fiecare dintre cele 100 de case, calculam contributiile si vedem factorul cel mai important (ponderea maxima)
            dominant_features = []
            features_list = self.backend.feature_names
            
            for idx, row in test_subset.iterrows():
                _, _, contribs = self.backend.get_local_contributions(row.to_dict())
                # Gasim caracteristica cu contributia absoluta maxima
                max_feat = max(contribs.keys(), key=lambda k: abs(contribs[k]))
                dominant_features.append(max_feat)
                
            self.plot_categorical_scatter(lons, lats, dominant_features, "Model ML")
            
        elif color_option == "Factor Dominant (LLM Estimare)":
            # Obtinem factorul dominant indicat de LLM (folosind simulatorul rapid pentru harta)
            dominant_features = []
            for idx, row in test_subset.iterrows():
                _, _, contribs = self.backend.get_local_contributions(row.to_dict())
                # Pentru harta generam in mod simulat deoarece apelarea a 100 de instante Ollama ar dura minute bune
                explanation, _ = self.backend.generate_explanation(row.to_dict(), 0.0, contribs, force_simulation=True)
                parsed = self.backend.parse_llm_explanation(explanation)
                llm_w = parsed["llm_weights"]
                max_feat = max(llm_w.keys(), key=lambda k: llm_w[k])
                dominant_features.append(max_feat)
                
            self.plot_categorical_scatter(lons, lats, dominant_features, "LLM Estimare")

        self.ax_map.set_xlabel("Longitudine", color=self.fg_color, fontsize=8)
        self.ax_map.set_ylabel("Latitudine", color=self.fg_color, fontsize=8)
        self.ax_map.tick_params(colors=self.fg_color, labelsize=8)
        self.fig_map.tight_layout()
        self.canvas_map.draw()

    def plot_categorical_scatter(self, lons, lats, categories, source_name):
        feature_colors = {
            "MedInc": "#f87171",     # Rosu deschis
            "HouseAge": "#60a5fa",    # Albastru
            "AveRooms": "#34d399",    # Verde
            "AveBedrms": "#fbbf24",   # Portocaliu
            "Population": "#c084fc",  # Violet
            "AveOccup": "#f472b6",    # Roz
            "Latitude": "#2dd4bf",    # Teal
            "Longitude": "#22d3ee"    # Cyan
        }
        
        unique_cats = sorted(list(set(categories)))
        
        for cat in unique_cats:
            indices = [i for i, c in enumerate(categories) if c == cat]
            cat_lons = lons[indices]
            cat_lats = lats[indices]
            
            color = feature_colors.get(cat, "#9ca3af")
            label = FEATURE_MAP[cat]["ro"]
            
            self.ax_map.scatter(cat_lons, cat_lats, color=color, label=label, s=40, alpha=0.85, edgecolor="none")
            
        self.ax_map.legend(facecolor='#273142', edgecolor='none', labelcolor=self.fg_color, fontsize=8, loc="upper right")
        self.ax_map.set_title(f"Harta Spatiala XAI: Factorul Local cel mai Influent ({source_name})", color=self.fg_color, fontsize=10)

    # ----------------- TAB 3: EVALUARE BATCH SI STATISTICI -----------------
    def setup_tab3(self):
        batch_control = ttk.Frame(self.tab3, style="Card.TFrame")
        batch_control.pack(fill="x", padx=10, pady=5)
        
        inner_batch = ttk.Frame(batch_control, style="Card.TFrame")
        inner_batch.pack(fill="both", padx=10, pady=8)
        
        ttk.Label(inner_batch, text="Evaluare Batch Completa si Analiza Distributiei", style="Header.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 5))
        
        ttk.Label(inner_batch, text="Numar locatii pentru testare batch:", style="CardText.TLabel").grid(row=1, column=0, sticky="w")
        self.spin_batch_size = ttk.Spinbox(inner_batch, from_=10, to=50, width=5)
        self.spin_batch_size.set(30)
        self.spin_batch_size.grid(row=1, column=1, sticky="w", padx=5)
        
        self.btn_batch = ttk.Button(inner_batch, text="Ruleaza Evaluare Batch", command=self.run_batch_evaluation)
        self.btn_batch.grid(row=1, column=2, padx=15)
        
        self.lbl_batch_status = ttk.Label(inner_batch, text="Stare: In asteptare", style="CardText.TLabel")
        self.lbl_batch_status.grid(row=1, column=3, sticky="e")
        
        # ScrolledText pentru loguri si rezultate batch
        self.txt_batch_logs = scrolledtext.ScrolledText(self.tab3, height=20, width=115, bg="#1e2530", fg="#f3f4f6", relief="flat", font=('Courier', 9))
        self.txt_batch_logs.pack(fill="both", expand=True, padx=10, pady=5)

    def run_batch_evaluation(self):
        try:
            size = int(self.spin_batch_size.get())
        except ValueError:
            messagebox.showerror("Eroare", "Numarul introdus este invalid.")
            return
            
        self.btn_batch.config(state="disabled")
        self.lbl_batch_status.config(text="Stare: Se calculeaza...")
        self.root.update()
        
        try:
            force_sim = self.force_sim_var.get()
            
            def update_cb(curr, tot):
                self.lbl_batch_status.config(text=f"Stare: {curr}/{tot} case procesate")
                self.root.update()
                
            summary, logs = self.backend.run_batch_evaluation(
                batch_size=size,
                force_simulation=force_sim,
                progress_callback=update_cb
            )
            
            # Afisare rezultate
            self.txt_batch_logs.delete("1.0", tk.END)
            self.txt_batch_logs.insert(tk.END, "RAPORT EVALUARE BATCH:\n")
            self.txt_batch_logs.insert(tk.END, "="*95 + "\n")
            self.txt_batch_logs.insert(tk.END, f"Numar total case analizate:       {summary['batch_size']}\n")
            self.txt_batch_logs.insert(tk.END, f"Consistenta Calitativa Medie:     {summary['avg_consistency_rate']:.1f}%\n")
            self.txt_batch_logs.insert(tk.END, f"Recall Factor Primar Mediu:       {summary['avg_primary_recall_rate']:.1f}%\n")
            self.txt_batch_logs.insert(tk.END, f"Consistenta Distributie Ponderi:  {summary['avg_dist_consistency']:.1f}%\n")
            self.txt_batch_logs.insert(tk.END, f"Eroare Absoluta Medie Pret LLM:   {summary['avg_llm_prediction_error']:.1f}%\n")
            self.txt_batch_logs.insert(tk.END, f"Medie halucinatii per locuinta:   {summary['avg_hallucinations_per_house']:.2f}\n")
            self.txt_batch_logs.insert(tk.END, "="*95 + "\n\n")
            
            self.txt_batch_logs.insert(tk.END, f"{'ID':<6} | {'Pret Real':<11} | {'Pret Model':<11} | {'Pret LLM':<11} | {'Eroare LLM':<10} | {'Consist. (TVD)':<14} | {'Recall':<6} | {'Halucinatii'}\n")
            self.txt_batch_logs.insert(tk.END, "-"*105 + "\n")
            
            for l in logs:
                rec_str = "DA" if l["primary_recalled"] else "NU"
                hall_str = ", ".join([FEATURE_MAP[f]["ro"][:10] for f in l["hallucinations"]]) if l["hallucinations"] else "Niciuna"
                llm_p_str = f"${l['llm_prediction']:,.0f}" if l['llm_prediction'] > 0 else "N/A"
                
                self.txt_batch_logs.insert(tk.END, f"{l['house_id']:<6} | ${l['actual_price']:<10,.0f} | ${l['predicted_price']:<10,.0f} | {llm_p_str:<11} | {l['llm_pe_str']:<10} | {l['distribution_consistency']:<13.1f}% | {rec_str:<6} | {hall_str}\n")
                
            self.lbl_batch_status.config(text="Stare: Finalizat")
            
        except Exception as e:
            self.lbl_batch_status.config(text="Stare: Eroare")
            messagebox.showerror("Eroare", f"Nu s-a putut rula evaluarea: {str(e)}")
        finally:
            self.btn_batch.config(state="normal")

    # --------- TAB 4: BLEU/ROUGE - CALITATE TEXT LLM ---------
    def setup_tab4(self):
        header_frame = ttk.Frame(self.tab4, style="Card.TFrame")
        header_frame.pack(fill="x", padx=10, pady=5)

        inner_header = ttk.Frame(header_frame, style="Card.TFrame")
        inner_header.pack(fill="both", padx=10, pady=8)

        ttk.Label(inner_header, text="Evaluare Calitate Explicatii LLM - BLEU si ROUGE Scores", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(inner_header, text="Selecteaza o casa din Tab 1, genereaza explicatia, apoi verifica metrici de text aici.", font=('Helvetica', 9), foreground="#9ca3af").pack(anchor="w")

        # Frame pentru continut
        content_frame = ttk.Frame(self.tab4, style="Card.TFrame")
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Stanga: Explicatii
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

        ttk.Label(right_frame, text="METRICI DE CALITATE:", style="Header.TLabel").pack(anchor="w", pady=(5, 10))

        # BLEU Score
        bleu_frame = ttk.Frame(right_frame, style="Card.TFrame")
        bleu_frame.pack(fill="x", pady=5)
        ttk.Label(bleu_frame, text="BLEU Score:", style="CardText.TLabel", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        self.lbl_bleu_tab4 = ttk.Label(bleu_frame, text="-- %", font=('Helvetica', 24, 'bold'), foreground="#60a5fa")
        self.lbl_bleu_tab4.pack(anchor="w")
        ttk.Label(bleu_frame, text="Overlap de cuvinte/fraze cu explicatia ideala", style="CardText.TLabel", font=('Helvetica', 8)).pack(anchor="w")

        # ROUGE Score
        rouge_frame = ttk.Frame(right_frame, style="Card.TFrame")
        rouge_frame.pack(fill="x", pady=5)
        ttk.Label(rouge_frame, text="ROUGE Score:", style="CardText.TLabel", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        self.lbl_rouge_tab4 = ttk.Label(rouge_frame, text="-- %", font=('Helvetica', 24, 'bold'), foreground="#f87171")
        self.lbl_rouge_tab4.pack(anchor="w")
        ttk.Label(rouge_frame, text="Asemanare semantica cu explicatia ideala", style="CardText.TLabel", font=('Helvetica', 8)).pack(anchor="w")

        # Interpretare
        interp_frame = ttk.Frame(right_frame, style="Card.TFrame")
        interp_frame.pack(fill="both", expand=True, pady=10)
        ttk.Label(interp_frame, text="INTERPRETARE:", style="Header.TLabel").pack(anchor="w", pady=(5, 8))
        self.txt_interpretation = scrolledtext.ScrolledText(interp_frame, height=8, width=35, bg="#1e2530", fg="#d1d5db", relief="flat", font=('Helvetica', 8), wrap=tk.WORD)
        self.txt_interpretation.pack(fill="both", expand=True)

    def update_bleu_rouge_dashboard(self, features, llm_explanation_text, contributions, eval_res):
        """Update Tab4 dashboard cu datele actuale din Tab1"""
        # Genereaza ideal explanation
        ideal_explanation = self.generate_ideal_explanation(features, eval_res["top_real_factors"])

        # Calculeaza BLEU si ROUGE
        bleu = self.calculate_bleu_score(ideal_explanation, llm_explanation_text)
        rouge = self.calculate_rouge_score(ideal_explanation, llm_explanation_text)

        # Actualizeaza text boxes
        self.txt_ideal_explanation.delete("1.0", tk.END)
        self.txt_ideal_explanation.insert(tk.END, ideal_explanation)

        llm_text_only = llm_explanation_text.split('[')[0].strip() if '[' in llm_explanation_text else llm_explanation_text[:300]
        self.txt_llm_explanation_tab4.delete("1.0", tk.END)
        self.txt_llm_explanation_tab4.insert(tk.END, llm_text_only)

        # Actualizeaza scores
        self.lbl_bleu_tab4.config(text=f"{bleu:.2f}%")
        self.lbl_rouge_tab4.config(text=f"{rouge:.2f}%")

        # Interpretare
        interp = f"BLEU: {bleu:.2f}%\n"
        if bleu < 5:
            interp += "LLM foloseste cuvinte diferite decat templatul ideal - Normal in limbaj natural.\n\n"
        else:
            interp += "LLM se aliniaza bine cu cuvintele cheie ale explicatiei ideale.\n\n"

        interp += f"ROUGE: {rouge:.2f}%\n"
        if rouge > 20:
            interp += "Asemanare semantica buna - LLM intelege conceptele relevante.\n\n"
        else:
            interp += "Asemanare semantica redusa - LLM poate discuta alte aspecte.\n\n"

        interp += f"Factori reali: {', '.join([FEATURE_MAP[f]['ro'][:15] for f in eval_res['top_real_factors'][:3]])}"

        self.txt_interpretation.delete("1.0", tk.END)
        self.txt_interpretation.insert(tk.END, interp)

    def generate_ideal_explanation(self, features, top_factors):
        """Genereaza explicatia ideala bazata pe factori reali"""
        top_3_labels = [FEATURE_MAP[f]['ro'] for f in top_factors[:3]]
        factor_text = ", ".join(top_3_labels)

        ideal = f"Valoarea proprietatii este condusa in principal de {factor_text}. "

        if top_factors[0] == "MedInc":
            ideal += f"Venitul mediu de ${features['MedInc']*10000:,.0f}/an este determinant puternic. "
        if any(f in top_factors for f in ["Latitude", "Longitude"]):
            ideal += f"Locatia la {features['Latitude']:.1f}°N, {features['Longitude']:.1f}°W impacteaza semnificativ. "
        if top_factors[0] in ["AveRooms", "AveBedrms"]:
            ideal += f"Dimensiunile proprietatii ({features['AveRooms']:.1f} camere) influienteaza pret. "

        return ideal

    def calculate_bleu_score(self, reference, generated, n=2):
        """Calculeaza BLEU score (n-gram match)"""
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

        return (matches / total) * 100 if total > 0 else 0.0

    def calculate_rouge_score(self, reference, generated):
        """Calculeaza ROUGE score (semantic overlap)"""
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()

        if len(ref_tokens) == 0 or len(gen_tokens) == 0:
            return 0.0

        # LCS-based scoring
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

if __name__ == "__main__":
    # Eliminam afisarea ferestrei goale la matplotlib in fundal
    plt.close('all')
    root = tk.Tk()
    app = HousingEvaluatorGUI(root)
    root.mainloop()
