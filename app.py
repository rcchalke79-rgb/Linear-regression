import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Linear.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HomeWorth AI · House Price Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --c1:#6a11cb; --c2:#2575fc; --c3:#00c9a7; --c4:#ff6a88;
    --ink:#1f2340; --muted:#6b7090; --card:rgba(255,255,255,.92);
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Poppins',sans-serif; color:var(--ink); min-height:100vh;
    background:linear-gradient(-45deg,#1b1464,#6a11cb,#2575fc,#00c9a7);
    background-size:400% 400%; animation:flow 18s ease infinite;
    padding:32px 16px 48px;
  }
  @keyframes flow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
  .wrap{max-width:1050px;margin:0 auto}
  header{text-align:center;color:#fff;margin-bottom:28px}
  header .logo{font-size:44px}
  header h1{font-size:2.3rem;font-weight:700;letter-spacing:.5px}
  header p{opacity:.85;font-weight:300;margin-top:6px}
  .grid{display:grid;grid-template-columns:1.5fr 1fr;gap:24px}
  @media(max-width:860px){.grid{grid-template-columns:1fr}}
  .card{
    background:var(--card);border-radius:22px;padding:28px;
    box-shadow:0 20px 50px rgba(15,10,60,.35);backdrop-filter:blur(8px);
  }
  .card h2{font-size:1.15rem;margin-bottom:18px;display:flex;align-items:center;gap:8px}
  .card h2::before{content:"";width:6px;height:22px;border-radius:4px;
    background:linear-gradient(var(--c1),var(--c2))}
  .fields{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:560px){.fields{grid-template-columns:1fr}}
  label{display:block;font-size:.8rem;font-weight:500;color:var(--muted);margin-bottom:6px}
  input{
    width:100%;padding:12px 14px;border-radius:12px;border:2px solid #e3e6f5;
    font:inherit;font-size:.95rem;color:var(--ink);background:#fff;transition:.2s;
  }
  input:focus{outline:none;border-color:var(--c2);box-shadow:0 0 0 4px rgba(37,117,252,.15)}
  .full{grid-column:1/-1}
  button{
    width:100%;margin-top:22px;padding:15px;border:0;border-radius:14px;cursor:pointer;
    font:inherit;font-weight:600;font-size:1.05rem;color:#fff;letter-spacing:.4px;
    background:linear-gradient(90deg,var(--c1),var(--c2),var(--c3));background-size:200% 100%;
    transition:.4s;box-shadow:0 10px 24px rgba(37,117,252,.4);
  }
  button:hover{background-position:100% 0;transform:translateY(-2px)}
  .result{
    display:flex;flex-direction:column;justify-content:center;align-items:center;
    text-align:center;min-height:100%;
  }
  .result .label{color:var(--muted);font-size:.85rem;text-transform:uppercase;letter-spacing:2px}
  .price{
    font-size:2.8rem;font-weight:700;margin:10px 0;
    background:linear-gradient(90deg,var(--c1),var(--c2),var(--c3));
    -webkit-background-clip:text;background-clip:text;color:transparent;
    word-break:break-word;
  }
  .badge{
    display:inline-block;padding:6px 16px;border-radius:30px;font-size:.8rem;color:#fff;
    background:linear-gradient(90deg,var(--c4),#ff9a5a);
  }
  .placeholder{font-size:3.4rem;opacity:.5}
  .error{color:#d6336c;font-weight:500}
  footer{text-align:center;color:#fff;opacity:.7;font-size:.8rem;margin-top:26px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">🏡</div>
    <h1>HomeWorth AI</h1>
    <p>Estimate a property's market value with a Linear Regression model</p>
  </header>

  <div class="grid">
    <div class="card">
      <h2>Property Details</h2>
      <form method="post" action="/">
        <div class="fields">
          <div>
            <label>Square Footage (sq ft)</label>
            <input type="number" step="any" name="Square_Footage" required
                   value="{{ v.get('Square_Footage', 2000) }}">
          </div>
          <div>
            <label>Year Built</label>
            <input type="number" name="Year_Built" required min="1800" max="2100"
                   value="{{ v.get('Year_Built', 2005) }}">
          </div>
          <div>
            <label>Bedrooms</label>
            <input type="number" name="Num_Bedrooms" required min="0"
                   value="{{ v.get('Num_Bedrooms', 3) }}">
          </div>
          <div>
            <label>Bathrooms</label>
            <input type="number" name="Num_Bathrooms" required min="0"
                   value="{{ v.get('Num_Bathrooms', 2) }}">
          </div>
          <div>
            <label>Lot Size</label>
            <input type="number" step="any" name="Lot_Size" required min="0"
                   value="{{ v.get('Lot_Size', 1.5) }}">
          </div>
          <div>
            <label>Garage Size (cars)</label>
            <input type="number" name="Garage_Size" required min="0"
                   value="{{ v.get('Garage_Size', 2) }}">
          </div>
          <div class="full">
            <label>Neighborhood Quality</label>
            <input type="number" step="any" name="Neighborhood_Quality" required
                   value="{{ v.get('Neighborhood_Quality', 7) }}">
          </div>
        </div>
        <button type="submit">✨ Predict Price</button>
      </form>
    </div>

    <div class="card result">
      {% if error %}
        <div class="placeholder">⚠️</div>
        <p class="error">{{ error }}</p>
      {% elif prediction is not none %}
        <span class="label">Estimated Price</span>
        <div class="price">${{ "{:,.0f}".format(prediction) }}</div>
        <span class="badge">Linear Regression Estimate</span>
      {% else %}
        <div class="placeholder">📊</div>
        <p style="color:var(--muted);margin-top:10px">
          Fill in the property details and press <b>Predict Price</b> to see the estimate.
        </p>
      {% endif %}
    </div>
  </div>

  <footer>Built with Flask &amp; scikit-learn · Deployed on Render</footer>
</div>
</body>
</html>
"""


def predict_price(values: dict) -> float:
    row = pd.DataFrame([[float(values[f]) for f in FEATURES]], columns=FEATURES)
    return float(model.predict(row)[0])


@app.route("/", methods=["GET", "POST"])
def home():
    prediction, error, values = None, None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            prediction = predict_price(values)
        except (KeyError, ValueError):
            error = "Please enter valid numbers in every field."
    return render_template_string(PAGE, prediction=prediction, error=error, v=values)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify({"predicted_price": round(predict_price(data), 2)})
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Required numeric fields: " + ", ".join(FEATURES)}), 400


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
