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
        self.baseline = None
        self.metrics = {}
        self.global_importances = {}
        self.is_trained = False

    def train_model(self, train_fraction=0.8):
        """Train Random Forest on California Housing, optionally subsampling training data."""
        data = fetch_california_housing(as_frame=True)
        self.feature_names = list(data.feature_names)

        X = data.data
        y = data.target * 100000.0  # Convert to USD

        # Split train/test (test always 20% for consistency)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Subsample training data if fraction < 1.0
        if train_fraction < 1.0:
            train_indices = self.X_train.sample(frac=train_fraction, random_state=42).index
            X_train = self.X_train.loc[train_indices]
            y_train = self.y_train.loc[train_indices]
        else:
            X_train = self.X_train
            y_train = self.y_train

        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            min_samples_split=5
        )
        self.model.fit(X_train, y_train)
        self.baseline = X_train.mean()

        # Compute global metrics
        y_pred = self.model.predict(self.X_test)
        r2 = self.model.score(self.X_test, self.y_test)
        mae = np.mean(np.abs(self.y_test.values - y_pred))
        rmse = np.sqrt(np.mean((self.y_test.values - y_pred) ** 2))

        self.metrics = {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "mean_price": float(y.mean()),
            "std_price": float(y.std()),
            "train_size": len(X_train),
            "test_size": len(self.X_test)
        }

        # Global feature importances
        importances = self.model.feature_importances_
        self.global_importances = {
            name: float(imp) for name, imp in zip(self.feature_names, importances)
        }
        self.global_importances = dict(
            sorted(self.global_importances.items(), key=lambda x: x[1], reverse=True)
        )

        self.is_trained = True
        return self.metrics, self.global_importances

    def get_test_houses(self, n=50):
        """Get n test samples with predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained.")

        indices = self.X_test.index[:n]
        houses = []

        for idx in indices:
            row = self.X_test.loc[idx]
            actual_price = float(self.y_test.loc[idx])
            pred_price = float(self.model.predict([row.values])[0])

            features_dict = {col: float(row[col]) for col in self.feature_names}

            houses.append({
                "id": int(idx),
                "features": features_dict,
                "actual_price": actual_price,
                "predicted_price": pred_price
            })

        return houses

    def get_local_contributions(self, features_dict):
        """Compute local SHAP-like contributions using perturbation."""
        if not self.is_trained:
            raise ValueError("Model not trained.")

        x = np.array([features_dict[name] for name in self.feature_names])
        y_pred = float(self.model.predict([x])[0])

        x_baseline = self.baseline.values
        y_baseline = float(self.model.predict([x_baseline])[0])

        # Perturbation: replace each feature with baseline, compute delta
        contributions = {}
        for i, name in enumerate(self.feature_names):
            x_perturbed = x.copy()
            x_perturbed[i] = x_baseline[i]
            y_perturbed = float(self.model.predict([x_perturbed])[0])
            contributions[name] = y_pred - y_perturbed

        # Sort by absolute contribution
        contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return y_pred, y_baseline, contributions

    def query_ollama(self, prompt, model_name="llama3.2"):
        """Query local Ollama instance."""
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "").strip(), False
            else:
                return "", True
        except Exception:
            return "", True

    def generate_simulated_explanation(self, features_dict, pred_price, contributions):
        """Generate plausible simulated explanation when Ollama unavailable."""
        top_features = list(contributions.keys())[:3]
        primary = top_features[0]

        # 70% consistent, 30% hallucinating
        is_consistent = random.random() < 0.7

        # Calculate simulated prediction first so it is consistent with the text
        sim_pred = pred_price * random.uniform(0.92, 1.08)

        intro = f"Această proprietate este estimată de ML la ${sim_pred:,.0f} deoarece "

        if is_consistent:
            reasons = []
            if primary in ["MedInc"]:
                val = features_dict["MedInc"]
                reasons.append("venitului mediu ridicat al zonei" if val > 5.0 else "venitului mediu scăzut")
            if primary in ["HouseAge"]:
                val = features_dict["HouseAge"]
                reasons.append("vechimei construcțiilor" if val > 35 else "modernității locuințelor")
            if primary in ["Latitude", "Longitude"]:
                reasons.append("poziției geografice favorabile în California")
            if not reasons:
                reasons.append("caracteristicilor demografice ale regiunii")

            text = intro + ", ".join(reasons) + "."
        else:
            # Hallucinate a minor factor
            halluc_feat = random.choice([f for f in top_features if f != primary])
            text = intro + f"presupusului impact major al {FEATURE_MAP[halluc_feat]['ro'].lower()}."

        # Generate weights (sum to 100)
        abs_contrib = {k: abs(v) for k, v in contributions.items()}
        total = sum(abs_contrib.values())

        if total > 0:
            weights = {k: (abs_contrib[k] / total) * 100 for k in self.feature_names}
        else:
            weights = {k: 100 / 8 for k in self.feature_names}

        # Shuffle weights if hallucinating
        if not is_consistent:
            sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            weights[sorted_w[0][0]], weights[sorted_w[-1][0]] = weights[sorted_w[-1][0]], weights[sorted_w[0][0]]

        # Normalize to integers summing to 100
        weights = {k: int(round(v)) for k, v in weights.items()}
        diff = 100 - sum(weights.values())
        weights[list(weights.keys())[0]] += diff

        # Structured block
        weights_block = " ".join([f"[{k.upper()}: {weights[k]}]" for k in self.feature_names])
        return f"{text}\n\n[PREDICTION: {sim_pred:.0f}]\n{weights_block}", True

    def generate_explanation(self, features_dict, pred_price, contributions, force_simulation=False):
        """Generate LLM explanation (or simulated if unavailable)."""
        prompt = f"""Analist imobiliar: casă California cu venit {features_dict['MedInc']:.1f}, vârstă {features_dict['HouseAge']:.0f}a, camere {features_dict['AveRooms']:.1f}, dormitoare {features_dict['AveBedrms']:.0f}, populație {int(features_dict['Population'])}, ocupare {features_dict['AveOccup']:.1f}.

Estimează preț (doar număr USD), explică 2-3 fraze simplu de ce, apoi ponderi 0-100% pentru fiecare dintre: MEDINC, HOUSEAGE, AVEROOMS, AVEBEDRMS, POPULATION, AVEOCCUP, LATITUDE, LONGITUDE (suma=100%).

Final: [PREDICTION: X] [MEDINC: %] [HOUSEAGE: %] [AVEROOMS: %] [AVEBEDRMS: %] [POPULATION: %] [AVEOCCUP: %] [LATITUDE: %] [LONGITUDE: %]

Răspunde doar în blocul structurat și explicație. Limba: română."""

        if force_simulation:
            return self.generate_simulated_explanation(features_dict, pred_price, contributions)

        response, is_sim = self.query_ollama(prompt)
        if response and not is_sim:
            return response, is_sim
        else:
            return self.generate_simulated_explanation(features_dict, pred_price, contributions)

    def parse_llm_explanation(self, text):
        """Extract keywords (keyword matching) and structured data (PREDICTION, weights)."""
        text_lower = text.lower()
        detected_features = set()

        # Keyword matching
        for feature, keywords in KEYWORD_DICTIONARY.items():
            if any(kw in text_lower for kw in keywords):
                detected_features.add(feature)

        # Parse structured block
        llm_pred = None
        llm_weights = {name: 0.0 for name in self.feature_names}

        # Extract prediction
        pred_match = re.search(r'\[PREDICTION:\s*([\d\.,\s\$]+)\]', text, re.IGNORECASE)
        if pred_match:
            pred_str = pred_match.group(1).replace('$', '').replace(',', '').replace(' ', '')
            try:
                llm_pred = float(pred_str)
            except ValueError:
                pass

        # Extract weights
        weights_found = 0
        for name in self.feature_names:
            pattern = r'\[' + re.escape(name.upper()) + r':\s*(\d+)%?\]'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    llm_weights[name] = float(match.group(1))
                    weights_found += 1
                except ValueError:
                    pass

        # Normalize weights to percentages
        if weights_found > 0:
            total = sum(llm_weights.values())
            if total > 0:
                llm_weights = {k: (v / total) * 100 for k, v in llm_weights.items()}
                # Add high-weight features to detected (confidence boost)
                for k, v in llm_weights.items():
                    if v >= 10:
                        detected_features.add(k)
        else:
            # Fallback: uniform distribution
            llm_weights = {k: 100 / 8 for k in self.feature_names}

        return {
            "keywords_detected": list(detected_features),
            "llm_prediction": llm_pred,
            "llm_weights": llm_weights
        }

    def evaluate_explanation_consistency(self, true_contributions, parsed_data, top_k=3):
        """Evaluate factual (keyword recall) and distributional (weight similarity) consistency."""

        # Get top real factors
        sorted_real = sorted(
            true_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_real_factors = [x[0] for x in sorted_real[:top_k]]
        primary_real = sorted_real[0][0]

        llm_factors = parsed_data["keywords_detected"]
        llm_weights = parsed_data["llm_weights"]
        llm_pred = parsed_data["llm_prediction"]

        # 1. Factual consistency: % of LLM mentions that are in top-K real factors
        if llm_factors:
            correct = [f for f in llm_factors if f in top_real_factors]
            factual_consistency = len(correct) / len(llm_factors)

            # Primary recall: did LLM mention the dominant real factor?
            primary_recalled = primary_real in llm_factors

            hallucinations = [f for f in llm_factors if f not in top_real_factors]
            omissions = [f for f in top_real_factors if f not in llm_factors]
        else:
            factual_consistency = 0.0
            primary_recalled = False
            hallucinations = []
            omissions = top_real_factors

        # 2. Weight distribution consistency (L1-based)
        abs_real = {k: abs(v) for k, v in true_contributions.items()}
        total = sum(abs_real.values())

        if total > 0:
            real_weights = {k: (abs_real[k] / total) * 100 for k in self.feature_names}
        else:
            real_weights = {k: 100 / 8 for k in self.feature_names}

        l1_distance = sum(abs(real_weights[k] - llm_weights[k]) for k in self.feature_names)
        dist_consistency = 100 - (0.5 * l1_distance)  # L1 sim: 100 - 0.5*L1_dist

        return {
            "factual_consistency": factual_consistency,
            "primary_recalled": primary_recalled,
            "hallucinations": hallucinations,
            "omissions": omissions,
            "top_real_factors": top_real_factors,
            "primary_real": primary_real,
            "distribution_consistency": dist_consistency,
            "real_weights": real_weights,
            "llm_weights": llm_weights,
            "llm_prediction": llm_pred
        }

    def run_batch_evaluation(self, batch_size=30, force_simulation=False, progress_callback=None):
        """Evaluate LLM consistency on batch_size test samples."""
        if not self.is_trained:
            raise ValueError("Model not trained.")

        test_subset = self.X_test.head(batch_size)
        y_subset = self.y_test.head(batch_size)

        logs = []
        totals = {
            "consistency": 0.0,
            "recall": 0.0,
            "dist": 0.0,
            "hallucinations": 0,
            "valid_llm_preds": 0,
            "llm_pe": 0.0
        }

        for idx, (db_idx, row) in enumerate(test_subset.iterrows()):
            features = row.to_dict()
            actual = float(y_subset.loc[db_idx])

            pred, base, contribs = self.get_local_contributions(features)
            explanation, was_sim = self.generate_explanation(features, pred, contribs, force_simulation)

            parsed = self.parse_llm_explanation(explanation)
            eval_res = self.evaluate_explanation_consistency(contribs, parsed)

            totals["consistency"] += eval_res["factual_consistency"]
            totals["recall"] += (1.0 if eval_res["primary_recalled"] else 0.0)
            totals["dist"] += eval_res["distribution_consistency"]
            totals["hallucinations"] += len(eval_res["hallucinations"])

            llm_pred = eval_res["llm_prediction"]
            pe_str = "N/A"
            if llm_pred and llm_pred > 0:
                pe = abs(llm_pred - actual) / actual * 100
                totals["llm_pe"] += pe
                totals["valid_llm_preds"] += 1
                pe_str = f"{pe:.1f}%"

            logs.append({
                "house_id": int(db_idx),
                "actual_price": actual,
                "predicted_price": pred,
                "llm_prediction": llm_pred if llm_pred else 0.0,
                "llm_pe_str": pe_str,
                "top_real_factors": eval_res["top_real_factors"],
                "llm_mentioned_factors": parsed["keywords_detected"],
                "consistency_score": eval_res["factual_consistency"] * 100,
                "distribution_consistency": eval_res["distribution_consistency"],
                "primary_recalled": eval_res["primary_recalled"],
                "hallucinations": eval_res["hallucinations"],
                "is_simulated": was_sim
            })

            if progress_callback:
                progress_callback(idx + 1, batch_size)

        # Compute averages
        summary = {
            "batch_size": batch_size,
            "avg_consistency_rate": (totals["consistency"] / batch_size) * 100,
            "avg_primary_recall_rate": (totals["recall"] / batch_size) * 100,
            "avg_dist_consistency": totals["dist"] / batch_size,
            "avg_llm_prediction_error": (totals["llm_pe"] / totals["valid_llm_preds"]) if totals["valid_llm_preds"] > 0 else 0.0,
            "avg_hallucinations_per_house": totals["hallucinations"] / batch_size,
            "total_hallucinations_count": totals["hallucinations"]
        }

        return summary, logs
