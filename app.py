import pandas as pd
import re
import math
from collections import Counter
from flask import Flask, request, jsonify, render_template

# ====================== EVALUATION IMPORTS ======================
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# =============================================================================
# 1. LOAD REAL-TIME DATASET (UCI SMSSpamCollection)
# =============================================================================
try:
    df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'], encoding='utf-8')
    
    SPAM_MESSAGES = df[df['label'] == 'spam']['message'].tolist()
    HAM_MESSAGES = df[df['label'] == 'ham']['message'].tolist()
    print(f"--- Dataset Loaded: {len(df)} messages found ---")
except FileNotFoundError:
    print("CRITICAL ERROR: 'SMSSpamCollection' not found. Please place the file in this folder.")
    SPAM_MESSAGES, HAM_MESSAGES = [], []

# =============================================================================
# 2. TOKENIZER & VOCABULARY BUILDING
# =============================================================================
def tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return [t for t in text.split() if t]

# Process the entire dataset
all_spam_tokens = [t for msg in SPAM_MESSAGES for t in tokenize(msg)]
all_ham_tokens = [t for msg in HAM_MESSAGES for t in tokenize(msg)]

spam_freq = Counter(all_spam_tokens)
ham_freq = Counter(all_ham_tokens)

S = len(all_spam_tokens)
H = len(all_ham_tokens)
vocab = set(spam_freq.keys()) | set(ham_freq.keys())
V = len(vocab) 

# =============================================================================
# 3. CORE BAYESIAN FUNCTIONS
# =============================================================================
def p_word_given_spam(word: str) -> float:
    return (spam_freq.get(word, 0) + 1) / (S + V)

def p_word_given_ham(word: str) -> float:
    return (ham_freq.get(word, 0) + 1) / (H + V)

# =============================================================================
# 4. CLASSIFIER (FIXED - handles underflow safely)
# =============================================================================
def classify(message: str) -> dict:
    tokens = tokenize(message)
    if not tokens:
        return {"error": "empty", "spam_pct": 50.0}

    log_spam = math.log(0.5)
    log_ham = math.log(0.5)
    word_data = []

    for word in tokens:
        ps = p_word_given_spam(word)
        ph = p_word_given_ham(word)
        
        log_spam += math.log(ps)
        log_ham += math.log(ph)

        spam_count = spam_freq.get(word, 0)
        ham_count  = ham_freq.get(word, 0)
        
        p_spam_given_word = round(100 * ps / (ps + ph), 2) if (ps + ph) > 0 else 50.0

        word_data.append({
            "word": word,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "p_word_spam": round(ps, 6),
            "p_word_ham": round(ph, 6),
            "p_spam_given_word": p_spam_given_word,
            "is_known": word in vocab
        })

    # FIXED: Safe conversion from log-space
    try:
        raw_spam = math.exp(log_spam)
        raw_ham  = math.exp(log_ham)
        total = raw_spam + raw_ham
        if total == 0:  # underflow case
            spam_pct = 100.0 if log_spam > log_ham else 0.0
        else:
            spam_pct = (raw_spam / total) * 100
    except OverflowError:
        spam_pct = 100.0 if log_spam > log_ham else 0.0

    return {
        "tokens": tokens,
        "spam_pct": round(spam_pct, 2),
        "word_data": word_data,
        "log_spam": round(log_spam, 6),
        "log_ham": round(log_ham, 6),
        "raw_spam": round(raw_spam, 8) if 'raw_spam' in locals() else 0,
        "raw_ham": round(raw_ham, 8) if 'raw_ham' in locals() else 0,
        "complexity": "O(n)"
    }

# =============================================================================
# 5. FLASK ROUTES
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify_route():
    data = request.get_json() or {}
    message = data.get("message", "")
    return jsonify(classify(message))

# =============================================================================
# EVALUATION ROUTE - Train/Test Split + Metrics (required by assignment)
# =============================================================================
@app.route("/evaluate", methods=["GET"])
def evaluate():
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        result = classify(row['message'])
        pred = "spam" if result["spam_pct"] >= 50 else "ham"
        y_true.append(row['label'])
        y_pred.append(pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="spam")
    rec = recall_score(y_true, y_pred, pos_label="spam")
    f1 = f1_score(y_true, y_pred, pos_label="spam")
    
    return jsonify({
        "accuracy": round(acc*100, 2),
        "precision": round(prec*100, 2),
        "recall": round(rec*100, 2),
        "f1": round(f1*100, 2),
        "confusion_matrix": cm.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)