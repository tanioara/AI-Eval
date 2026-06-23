import numpy as np
import pandas as pd
import requests
import random
import re
import os
from collections import Counter
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


def calculate_bleu_score(reference_text, generated_text, n=2):
    """Calculate BLEU score (n-gram precision). Higher = more similar to reference."""
    ref_tokens = reference_text.lower().split()
    gen_tokens = generated_text.lower().split()

    if len(gen_tokens) == 0:
        return 0.0

    # Calculate n-gram matches
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

    return (matches / total) * 100 if total > 0 else 0.0


def calculate_rouge_score(reference_text, generated_text):
    """Calculate ROUGE-L score (longest common subsequence). Higher = better overlap."""
    ref_tokens = reference_text.lower().split()
    gen_tokens = generated_text.lower().split()

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

        # Train Random Forest - optimized for accuracy
        self.model = RandomForestRegressor(
            n_estimators=500,          # More trees = better accuracy
            max_depth=20,              # Deeper trees capture more patterns
            min_samples_split=2,       # Allow granular splits
            min_samples_leaf=1,        # Fine-grained leaves
            random_state=42,
            n_jobs=-1,
            verbose=0
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
            "options": {"temperature": 0.2, "top_p": 0.85, "num_predict": 400}
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                if result:
                    return result, False
            return "", True
        except requests.exceptions.Timeout:
            return "", True
        except Exception as e:
            return "", True

    def query_groq(self, prompt, model_name="llama-3.1-70b-versatile"):
        """Query Groq API (free tier, fast inference)."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "", True

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                return content.strip(), False
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

    def generate_explanation(self, features_dict, pred_price, contributions, force_simulation=False, use_ollama=True):
        """Generate LLM explanation (Ollama or Groq, fallback to simulated)."""

        if force_simulation:
            return self.generate_simulated_explanation(features_dict, pred_price, contributions)

        # Try Ollama first (if use_ollama=True)
        if use_ollama:
            ollama_prompt = f"""ESTIMATE CALIFORNIA HOUSE PRICE FROM FEATURES ONLY

You are a real estate analyst. Based ONLY on these house characteristics, estimate the market price.
Do NOT reference any pre-existing price - calculate independently from the data provided.

=== HOUSE CHARACTERISTICS (REAL DATA) ===
Median Area Income: ${features_dict['MedInc']*10000:,.0f}/year
House Age: {features_dict['HouseAge']:.0f} years
Average Rooms: {features_dict['AveRooms']:.2f}
Average Bedrooms: {features_dict['AveBedrms']:.2f}
Area Population: {int(features_dict['Population'])} people
Average Occupancy: {features_dict['AveOccup']:.1f} per household
Latitude: {features_dict['Latitude']:.2f}°N (California ranges 32-42°N)
Longitude: {features_dict['Longitude']:.2f}°W (California ranges -124 to -114°W)

=== MARKET CONTEXT ===
California house prices: $15,000 minimum to $500,000 maximum
Key value drivers:
- INCOME (45-50%): Wealthier areas = more expensive homes. Strong correlation.
- ROOMS/SIZE (20-25%): More rooms = higher value
- AGE (10-15%): Newer = more valuable
- LOCATION (10-20%): Coastal (lon ~-122°W, lat ~37°N) has premium. Inland standard.
- POPULATION (5-10%): Moderate density = stable; very high density mixed effects

=== YOUR TASK ===
1. Analyze each feature and its market impact
2. Calculate a realistic price estimate (use the features as your data source)
3. Explain your reasoning in 4-5 sentences (reference specific numbers from the data)
4. Assign percentage weights to each factor (MUST SUM TO 100%)
5. Provide your final price estimate

=== OUTPUT FORMAT (MUST FOLLOW EXACTLY) ===

[Explanation: 4-5 sentences analyzing the house characteristics and their impact on price]

[MEDINC: ___] [HOUSEAGE: ___] [AVEROOMS: ___] [AVEBEDRMS: ___] [POPULATION: ___] [AVEOCCUP: ___] [LATITUDE: ___] [LONGITUDE: ___]

[PREDICTION: $______]

=== IMPORTANT ===
Weights MUST be numbers ONLY (0-100), NOT dollar amounts.
Example: [MEDINC: 47] NOT [MEDINC: $47,000]

=== EXAMPLE ANALYSIS ===
This California property is in an area with median income of $65,000/year. The house is 18 years old. It has 6.1 rooms and 1.0 bedrooms. Location at 37.5°N, -122°W is Bay Area coast premium.

[MEDINC: 47] [HOUSEAGE: 12] [AVEROOMS: 25] [AVEBEDRMS: 8] [POPULATION: 5] [AVEOCCUP: 2] [LATITUDE: 1] [LONGITUDE: 0]

[PREDICTION: $285000]

=== YOUR RESPONSE ===
Explain in 2-3 sentences. Then EXACTLY 8 numbers (sum to 100). Then price.
Do NOT use dollar signs in weights. Only: [MEDINC: 45] etc."""

            response, is_sim = self.query_ollama(ollama_prompt)
            if response and not is_sim:
                return response, is_sim

        # Fallback to Groq
        groq_prompt = f"""You are a real estate appraiser. Estimate the market price of this California house based on its features ALONE. Make your own calculation - don't copy references.

HOUSE FEATURES:
- Median household income: ${features_dict['MedInc']*10000:.0f}/year
- House age: {features_dict['HouseAge']:.0f} years
- Average rooms: {features_dict['AveRooms']:.2f}
- Average bedrooms: {features_dict['AveBedrms']:.2f}
- Area population: {int(features_dict['Population'])}
- Average occupancy: {features_dict['AveOccup']:.1f} people
- Latitude: {features_dict['Latitude']:.2f}° (CA: 32-42°N)
- Longitude: {features_dict['Longitude']:.2f}° (CA: -124 to -114°W)

CONTEXT: California house prices: $15K-$500K

RESPOND WITH:
1. Explain your price estimate (3-4 sentences with calculation logic)
2. Factor weights (0-100%, sum=100%): [MEDINC: N] [HOUSEAGE: N] [AVEROOMS: N] [AVEBEDRMS: N] [POPULATION: N] [AVEOCCUP: N] [LATITUDE: N] [LONGITUDE: N]
3. Your estimate: [PREDICTION: $X]"""

        response, is_sim = self.query_groq(groq_prompt)
        if response and not is_sim:
            return response, is_sim

        # Last resort: simulation
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

        # Extract prediction - MUST have [PREDICTION: ...] format
        pred_match = re.search(r'\[PREDICTION:\s*\$?([\d,]+(?:\.\d+)?)\]', text, re.IGNORECASE)
        if pred_match:
            pred_str = pred_match.group(1).replace(',', '')
            try:
                llm_pred = float(pred_str)
                # Sanity check: price should be between $15K and $500K
                if llm_pred < 15000 or llm_pred > 500000:
                    llm_pred = None
            except ValueError:
                pass

        # Extract weights - search for [WORD: NUM%] pattern
        weights_found = 0
        for name in self.feature_names:
            # Match both [NAME: %] and [NAME: N%] formats
            pattern = r'\[' + re.escape(name.upper()) + r':\s*(\d+)\s*%?\s*\]'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1))
                    # Cap at 100 to avoid inflated weights
                    llm_weights[name] = min(val, 100)
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

    def generate_ideal_explanation(self, features_dict, real_factors, real_weights):
        """Generate ideal explanation based on real factors for comparison."""
        top_3 = real_factors[:3]
        factor_text = ", ".join([FEATURE_MAP[f]['en'] for f in top_3])

        ideal = f"This property's value is primarily driven by {factor_text}. "

        if real_factors[0] == "MedInc":
            ideal += f"The median income of ${features_dict['MedInc']*10000:,.0f}/year is a strong determinant. "
        if "Latitude" in real_factors or "Longitude" in real_factors:
            ideal += f"Location at {features_dict['Latitude']:.1f}°N, {features_dict['Longitude']:.1f}°W significantly impacts the valuation. "
        if real_factors[0] in ["AveRooms", "AveBedrms"]:
            ideal += f"The property has {features_dict['AveRooms']:.1f} rooms which affects the price. "
        if real_factors[0] == "HouseAge":
            ideal += f"The age of {features_dict['HouseAge']:.0f} years is a critical factor in determining market value."

        return ideal

    def run_batch_evaluation(self, batch_size=30, force_simulation=False, progress_callback=None):
        """Evaluate LLM consistency on batch_size test samples with BLEU/ROUGE metrics."""
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
            "llm_pe": 0.0,
            "bleu": 0.0,
            "rouge": 0.0
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

            # Calculate text quality metrics (BLEU & ROUGE)
            # Generate ideal explanation from real factors
            ideal_explanation = self.generate_ideal_explanation(features, eval_res["top_real_factors"], eval_res["real_weights"])

            bleu = calculate_bleu_score(ideal_explanation, explanation)
            rouge = calculate_rouge_score(ideal_explanation, explanation)

            totals["bleu"] += bleu
            totals["rouge"] += rouge

            # Extract explanation text (without prediction/weights tags)
            explanation_text = explanation.split('[')[0].strip() if '[' in explanation else explanation[:200]

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
                "is_simulated": was_sim,
                "bleu_score": bleu,
                "rouge_score": rouge,
                "ideal_explanation": ideal_explanation,
                "llm_explanation_text": explanation_text
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
            "total_hallucinations_count": totals["hallucinations"],
            "avg_bleu_score": totals["bleu"] / batch_size,
            "avg_rouge_score": totals["rouge"] / batch_size
        }

        return summary, logs
