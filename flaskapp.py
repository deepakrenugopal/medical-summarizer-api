from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load model once when app starts
model_path = "Deepakrenugopal/medical_summarizer"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

@app.route("/")
def home():
    return "Medical Summarization API is running!"

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()

    text = data.get("text", "")

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    summary_ids = model.generate(
        **inputs,
        max_length=64,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)