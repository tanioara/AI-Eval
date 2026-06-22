import numpy as np
import pandas as pd
import requests
import random
import re
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

FEATURE_MAP = {
    "MedInc": {"en": "Median Income", "ro": "Venit Median", "unit": "$10k", "desc": "Venitul median al gospodăriilor din zonă"},
    "HouseAge": {"en": "House Age", "ro": "Vârsta Casei", "unit": "ani", "desc": "Vârsta mediană a locuințelor din zonă"},
    "AveRooms": {"en": "Average Rooms", "ro": "Camere Medii", "unit": "camere", "desc": "Numărul mediu de camere per locuință"},
    "AveBedrms": {"en": "Average Bedrooms", "ro": "Dormitoare Medii", "unit": "dormitoare", "desc": "Numărul mediu de dormitoare per locuință"},
    "Population": {"en": "Population", "ro": "Populație", "unit": "locuitori", "desc": "Populația totală a zonei (block group)"},
    "AveOccup": {"en": "Average Occupancy", "ro": "Ocupanți Medii", "unit": "persoane", "desc": "Numărul mediu de ocupanți per gospodărie"},
    "Latitude": {"en": "Latitude", "ro": "Latitudine", "unit": "grade", "desc": "Coordonata geografică latitudine"},
    "Longitude": {"en": "Longitude", "ro": "Longitudine", "unit": "grade", "desc": "Coordonata geografică longitudine"}
}

KEYWORD_DICTIONARY = {
    "MedInc": ["venit", "salari", "financia", "incom", "wealth", "boga", "social", "earn", "bani", "starea material", "situație material"],
    "HouseAge": ["vech", "vârst", "veche", "nou", "construct", "ani", "age", "old", "new", "built", "istori", "timp", "durabilitate"],
    "AveRooms": ["camer", "spați", "generos", "room", "space", "dimension", "mărim", "suprafa", "compartiment"],
    "AveBedrms": ["dormito", "bedroom", "paturi", "pat "],
    "Population": ["popula", "locuito", "aglomer", "oameni", "densit", "peopl", "populat", "crowd", "reziden"],
    "AveOccup": ["ocupan", "membri", "famili", "gospodări", "occupan", "resid", "coabit", "persoane pe", "aglomerare", "înghesu"],
    "Latitude": ["locați", "poziți", "zon", "nord", "sud", "geograf", "apropier", "coordonat", "location", "area", "latitud", "hartă"],
    "Longitude": ["locați", "poziți", "zon", "est", "vest", "geograf", "apropier", "coordonat", "location", "area", "longitud", "ocean", "plaj", "coas", "californi", "litoral", "maritim", "apropiere de apă"]
}

class HousingEvaluatorBackend:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.baselines = None
        self.metrics = {}
        self.global_importances = {}
        self.is_trained = False
        self.train_fraction = 1.0
        
    def train_model(self, train_fraction=1.0):
        """
        Încarcă California Housing și antrenează Random Forest Regressor pe o fracțiune specifică de date (ex. 10% - 90%).
        """
        self.train_fraction = train_fraction
        data = fetch_california_housing(as_frame=True)
        self.feature_names = list(data.feature_names)
        
        X = data.data
        y = data.target * 100000.0  # Convertim prețul în USD
        
        # Split train/test (test set constant la 20% pentru a compara corect)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Sub-eșantionare set de antrenare dacă fracțiunea este sub 1.0
        if train_fraction < 1.0:
            X_train_sub = self.X_train.sample(frac=train_fraction, random_state=42)
            y_train_sub = self.y_train.loc[X_train_sub.index]
        else:
            X_train_sub = self.X_train
            y_train_sub = self.y_train
            
        self.model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        self.model.fit(X_train_sub, y_train_sub)
        
        self.baselines = X_train_sub.mean()
        
        # Calcul performanță globală pe setul de test
        predictions = self.model.predict(self.X_test)
        mse = np.mean((self.y_test - predictions) ** 2)
        mae = np.mean(np.abs(self.y_test - predictions))
        r2 = self.model.score(self.X_test, self.y_test)
        
        self.metrics = {
            "r2": float(r2),
            "mse": float(mse),
            "mae": float(mae),
            "mean_price": float(y.mean()),
            "std_price": float(y.std()),
            "train_size": len(X_train_sub),
            "test_size": len(self.X_test)
        }
        
        # Importanță globală a caracteristicilor
        importances = self.model.feature_importances_
        self.global_importances = {
            name: float(imp) for name, imp in zip(self.feature_names, importances)
        }
        self.global_importances = dict(sorted(self.global_importances.items(), key=lambda item: item[1], reverse=True))
        
        self.is_trained = True
        return self.metrics, self.global_importances
        
    def get_test_houses(self, n=50):
        if not self.is_trained:
            raise ValueError("Modelul nu este antrenat încă.")
            
        indices = self.X_test.index[:n]
        houses = []
        for idx in indices:
            row = self.X_test.loc[idx]
            actual_price = float(self.y_test.loc[idx])
            pred_price = float(self.model.predict([row.values])[0])
            
            features_data = {}
            for col in self.feature_names:
                features_data[col] = float(row[col])
                
            houses.append({
                "id": int(idx),
                "features": features_data,
                "actual_price": actual_price,
                "predicted_price": pred_price
            })
        return houses

    def get_local_contributions(self, features_dict):
        if not self.is_trained:
            raise ValueError("Modelul nu este antrenat încă.")
            
        x = np.array([features_dict[name] for name in self.feature_names])
        y_pred = float(self.model.predict([x])[0])
        x_base = self.baselines.values
        y_base = float(self.model.predict([x_base])[0])
        
        contributions = {}
        for i, name in enumerate(self.feature_names):
            x_perturbed = x.copy()
            x_perturbed[i] = x_base[i]
            y_perturbed = float(self.model.predict([x_perturbed])[0])
            contributions[name] = y_pred - y_perturbed
            
        sorted_contributions = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True))
        return y_pred, y_base, sorted_contributions

    def query_ollama(self, prompt, model_name="llama3.2"):
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=12)
            if response.status_code == 200:
                return response.json().get("response", "").strip(), False
            else:
                return f"Eroare de la Ollama (HTTP {response.status_code}): {response.text}", True
        except Exception as e:
            return f"Eroare de conexiune la Ollama: {str(e)}", True

    def generate_simulated_explanation(self, features_dict, pred_price, top_factors):
        """
        Generează o explicație simulată credibilă ce include blocul structurat de predicții și procente.
        """
        is_consistent = random.random() < 0.7
        top_list = list(top_factors.keys())
        primary_factor = top_list[0]
        secondary_factor = top_list[1]
        
        intro = f"Proprietatea este evaluata la ${pred_price:,.0f} din cauza "
        
        if is_consistent:
            reasons = []
            if primary_factor == "MedInc" or secondary_factor == "MedInc":
                val = features_dict["MedInc"]
                reasons.append("venitului mediu ridicat din aceasta zona rezidentiala") if val > 5.0 else reasons.append("venitului mediu scazut din zona")
            if primary_factor == "HouseAge" or secondary_factor == "HouseAge":
                val = features_dict["HouseAge"]
                reasons.append("vechimii istorice a constructiilor") if val > 35 else reasons.append("constructiilor recente si moderne")
            if primary_factor in ["Latitude", "Longitude"] or secondary_factor in ["Latitude", "Longitude"]:
                reasons.append("amplasarii favorabile in regiunea de coasta a Californiei")
                
            if len(reasons) == 0:
                reasons.append("indicilor demografici stabili din vecinatate")
                
            text = intro + " si ".join(reasons) + ". Valoarea reflecta parametrii zonei."
        else:
            hallucinated_reasons = []
            least_important = top_list[-3:]
            random_least = random.choice(least_important)
            
            if random_least == "Population":
                hallucinated_reasons.append(f"populatiei ridicate de {int(features_dict['Population'])} locuitori")
            elif random_least == "AveBedrms":
                hallucinated_reasons.append("numarului mare de dormitoare")
            else:
                hallucinated_reasons.append("indicelui de ocupare pe gospodarie")
                
            hallucinated_reasons.append("costului ridicat al materialelor si proiectelor imobiliare planificate")
            text = intro + " si ".join(hallucinated_reasons) + ". Piata locala arata o cerere in crestere."

        # Generare procente simulate ce insumeaza 100%
        abs_contribs = {k: abs(v) for k, v in top_factors.items()}
        total_abs = sum(abs_contribs.values())
        
        if total_abs > 0:
            weights = {k: (v / total_abs) * 100.0 for k, v in abs_contribs.items()}
        else:
            weights = {k: 12.5 for k in self.feature_names}
            
        if not is_consistent:
            # Schimbam ponderea factorului principal cu una mica (simulam greseala LLM)
            sorted_feats = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            top_f = sorted_feats[0][0]
            least_f = sorted_feats[-1][0]
            weights[top_f], weights[least_f] = weights[least_f], weights[top_f]
            
        # Re-normalizare la exact 100%
        w_sum = sum(weights.values())
        weights = {k: int(round((v / w_sum) * 100.0)) for k, v in weights.items()}
        # Ajustam diferenta de rotunjire pe primul element
        diff = 100 - sum(weights.values())
        weights[list(weights.keys())[0]] += diff
        
        sim_llm_pred = pred_price * random.uniform(0.88, 1.12)
        
        # Adaugare bloc structurat curat
        block = f"\n\n[PREDICTION: {sim_llm_pred:.0f}]\n"
        block += " ".join([f"[{k.upper()}: {weights[k]}]" for k in self.feature_names])
        return text + block

    def generate_explanation(self, features_dict, pred_price, top_factors, model_name="llama3.2", force_simulation=False):
        """
        Metodă ce cere LLM-ului explicația în limbaj natural, propria lui predicție de preț și ponderile.
        """
        prompt = f"""
Esti un analist imobiliar. O casa are urmatoarele caracteristici:
- Venit median in zona (MedInc): {features_dict['MedInc']:.2f} (in zeci de mii de dolari/an)
- Varsta medie a locuintelor (HouseAge): {features_dict['HouseAge']:.1f} ani
- Numarul mediu de camere (AveRooms): {features_dict['AveRooms']:.2f}
- Numarul mediu de dormitoare (AveBedrms): {features_dict['AveBedrms']:.2f}
- Populatia zonei (Population): {int(features_dict['Population'])}
- Ocuparea medie a locuintei (AveOccup): {features_dict['AveOccup']:.2f}
- Coordonate: Latitudine {features_dict['Latitude']:.2f}, Longitudine {features_dict['Longitude']:.2f}

Modelul de regresie a prezis un pret de ${pred_price:,.0f}.
Sarcina ta:
1. Explica in 2-3 fraze simple de ce casa are acest nivel de pret.
2. Fa propria ta estimare a pretului casei (doar valoarea numerica, in USD).
3. Atribui un procent de importanta (numar intreg intre 0 si 100) pentru fiecare dintre cele 8 caracteristici de mai sus in functie de cat de mult au influentat estimarea ta. Suma acestor 8 procente trebuie sa fie exact 100.

La finalul explicatiei tale, adauga OBLIGATORIU urmatorul bloc structurat, completat cu valorile tale (inlocuieste X si procente cu numere):
[PREDICTION: X]
[MEDINC: Procent] [HOUSEAGE: Procent] [AVEROOMS: Procent] [AVEBEDRMS: Procent] [POPULATION: Procent] [AVEOCCUP: Procent] [LATITUDE: Procent] [LONGITUDE: Procent]

Nu folosi alte cuvinte in blocul structurat. Raspunde in limba romana.
"""
        if force_simulation:
            return self.generate_simulated_explanation(features_dict, pred_price, top_factors), True
            
        response, is_simulated = self.query_ollama(prompt, model_name)
        if is_simulated:
            return self.generate_simulated_explanation(features_dict, pred_price, top_factors), True
        return response, False

    def parse_llm_explanation(self, text):
        """
        Detectează factorii prin keyword matching și extrage datele structurate (predicție LLM și ponderi).
        """
        text_lower = text.lower()
        detected_features = []
        
        # 1. Keyword matching simplu ca fallback
        for feature, keywords in KEYWORD_DICTIONARY.items():
            for kw in keywords:
                if kw in text_lower:
                    detected_features.append(feature)
                    break
                    
        # 2. Parsare structurată
        llm_pred = None
        llm_weights = {name: 0.0 for name in self.feature_names}
        has_structured = False
        
        # Extrage predictia
        pred_match = re.search(r'\[PREDICTION:\s*([\d\.,\s\$]+)\]', text, re.IGNORECASE)
        if pred_match:
            pred_str = pred_match.group(1).replace('$', '').replace(',', '').replace(' ', '').strip()
            try:
                llm_pred = float(pred_str)
            except ValueError:
                pass
                
        # Extrage ponderile
        weights_found = 0
        for name in self.feature_names:
            pattern = r'\[' + re.escape(name) + r':\s*(\d+)%?\]'
            weight_match = re.search(pattern, text, re.IGNORECASE)
            if weight_match:
                try:
                    llm_weights[name] = float(weight_match.group(1))
                    weights_found += 1
                except ValueError:
                    pass
                    
        if weights_found > 0:
            has_structured = True
            total = sum(llm_weights.values())
            if total > 0:
                # Normalizare la procente
                llm_weights = {k: (v / total) * 100.0 for k, v in llm_weights.items()}
                # Daca o pondere este semnificativa (>10%), o adaugam la detected_features (imbunatateste keyword parsing)
                for k, v in llm_weights.items():
                    if v >= 10.0:
                        detected_features.append(k)
        else:
            # Daca nu s-au gasit ponderi structurate, mapam pe baza keyword matching-ului
            detected_unique = list(set(detected_features))
            if detected_unique:
                share = 100.0 / len(detected_unique)
                llm_weights = {k: (share if k in detected_unique else 0.0) for k in self.feature_names}
            else:
                llm_weights = {k: 12.5 for k in self.feature_names}
                
        return {
            "keywords_detected": list(set(detected_features)),
            "llm_prediction": llm_pred,
            "llm_weights": llm_weights,
            "has_structured": has_structured
        }

    def evaluate_explanation_consistency(self, true_contributions, parsed_data, top_k=3):
        """
        Evaluează consistența calitativă (top K factori) și cantitativă (distribuția ponderilor și eroare de predicție).
        """
        sorted_real = sorted(true_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        top_real_features = [item[0] for item in sorted_real[:top_k]]
        primary_real_feature = sorted_real[0][0]
        
        llm_factors = parsed_data["keywords_detected"]
        llm_weights = parsed_data["llm_weights"]
        llm_pred = parsed_data["llm_prediction"]
        
        # 1. Scorul de consistență pe cuvinte cheie
        if not llm_factors:
            factual_consistency = 0.0
            primary_recall = 0.0
            hallucinations = []
            omissions = top_real_features
        else:
            correct_mentions = [f for f in llm_factors if f in top_real_features]
            factual_consistency = len(correct_mentions) / len(llm_factors)
            
            primary_recall = 0.0
            if primary_real_feature in llm_factors:
                primary_recall = 1.0
            elif primary_real_feature in ["Latitude", "Longitude"] and ("Latitude" in llm_factors or "Longitude" in llm_factors):
                primary_recall = 1.0
                
            hallucinations = [f for f in llm_factors if f not in top_real_features]
            # Corectam dublurile de coordonate
            if "Latitude" in hallucinations and "Longitude" in top_real_features:
                hallucinations.remove("Latitude")
            if "Longitude" in hallucinations and "Latitude" in top_real_features:
                hallucinations.remove("Longitude")
                
            omissions = [f for f in top_real_features if f not in llm_factors]
            if "Latitude" in omissions and "Longitude" in llm_factors:
                omissions.remove("Latitude")
            if "Longitude" in omissions and "Latitude" in llm_factors:
                omissions.remove("Longitude")

        # 2. Scorul de consistență pe distribuția ponderilor (Total Variation Distance similarity)
        # Convertim contributiile locale in ponderi absolute care insumeaza 100%
        abs_real = {k: abs(v) for k, v in true_contributions.items()}
        total_real = sum(abs_real.values())
        if total_real > 0:
            real_weights = {k: (v / total_real) * 100.0 for k, v in abs_real.items()}
        else:
            real_weights = {k: 12.5 for k in self.feature_names}
            
        # L1 similarity: 100 - 0.5 * sum(|W_real - W_llm|)
        l1_diff = sum(abs(real_weights[k] - llm_weights[k]) for k in self.feature_names)
        distribution_consistency = 100.0 - (0.5 * l1_diff)
        
        return {
            "factual_consistency": factual_consistency,
            "primary_recall": primary_recall,
            "hallucinations": hallucinations,
            "omissions": omissions,
            "top_real_features": top_real_features,
            "primary_real": primary_real_feature,
            "distribution_consistency": distribution_consistency,
            "real_weights": real_weights,
            "llm_weights": llm_weights,
            "llm_prediction": llm_pred
        }

    def run_batch_evaluation(self, batch_size=30, model_name="llama3.2", force_simulation=False, progress_callback=None):
        if not self.is_trained:
            raise ValueError("Modelul nu este antrenat încă.")
            
        test_subset = self.X_test.head(batch_size)
        y_subset = self.y_test.head(batch_size)
        
        detailed_logs = []
        total_consistency = 0.0
        total_recall = 0.0
        total_dist_consistency = 0.0
        total_hallucinations = 0
        valid_llm_preds = 0
        total_llm_pe = 0.0 # Procentul erorii absolute a LLM-ului
        
        for idx, (db_idx, row) in enumerate(test_subset.iterrows()):
            features_dict = row.to_dict()
            actual_price = float(y_subset.loc[db_idx])
            
            pred_price, y_base, contributions = self.get_local_contributions(features_dict)
            explanation, used_sim = self.generate_explanation(
                features_dict, pred_price, contributions, model_name, force_simulation
            )
            
            parsed_data = self.parse_llm_explanation(explanation)
            eval_results = self.evaluate_explanation_consistency(contributions, parsed_data)
            
            total_consistency += eval_results["factual_consistency"]
            total_recall += eval_results["primary_recall"]
            total_dist_consistency += eval_results["distribution_consistency"]
            total_hallucinations += len(eval_results["hallucinations"])
            
            # Verificare eroare predictie LLM
            llm_pred = eval_results["llm_prediction"]
            pe_str = "N/A"
            if llm_pred is not None:
                pe = abs(llm_pred - actual_price) / actual_price * 100.0
                total_llm_pe += pe
                valid_llm_preds += 1
                pe_str = f"{pe:.1f}%"
                
            detailed_logs.append({
                "house_id": int(db_idx),
                "actual_price": actual_price,
                "predicted_price": pred_price,
                "llm_prediction": llm_pred if llm_pred is not None else 0.0,
                "llm_pe_str": pe_str,
                "explanation": explanation,
                "top_real_factors": eval_results["top_real_features"],
                "llm_mentioned_factors": parsed_data["keywords_detected"],
                "consistency_score": eval_results["factual_consistency"] * 100.0,
                "distribution_consistency": eval_results["distribution_consistency"],
                "primary_recalled": bool(eval_results["primary_recall"]),
                "hallucinations": eval_results["hallucinations"],
                "is_simulated": used_sim
            })
            
            if progress_callback:
                progress_callback(idx + 1, batch_size)
                
        avg_consistency = total_consistency / batch_size
        avg_recall = total_recall / batch_size
        avg_dist_consistency = total_dist_consistency / batch_size
        avg_hallucinations_per_house = total_hallucinations / batch_size
        avg_llm_pe = (total_llm_pe / valid_llm_preds) if valid_llm_preds > 0 else None
        
        summary_stats = {
            "batch_size": batch_size,
            "avg_consistency_rate": float(avg_consistency * 100.0),
            "avg_primary_recall_rate": float(avg_recall * 100.0),
            "avg_dist_consistency": float(avg_dist_consistency),
            "avg_llm_prediction_error": float(avg_llm_pe) if avg_llm_pe is not None else 0.0,
            "avg_hallucinations_per_house": float(avg_hallucinations_per_house),
            "total_hallucinations_count": int(total_hallucinations)
        }
        
        return summary_stats, detailed_logs
