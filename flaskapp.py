from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

app = Flask(__name__)

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        model_path = os.environ.get("MODEL_PATH", "t5-small")  
        

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        model.to("cpu")  # force CPU (important for cloud)

@app.route("/")
def home():
    return "Medical Summarization API is running!"

@app.route("/load")
def load():
    load_model()
    return "Model loaded successfully!"

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        load_model()

        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data.get("findings")
        if not text:
        return jsonify({"error": "Missing 'findings' field"}), 400

        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        summary_ids = model.generate(
            **inputs,
            max_length=60, 
            min_length=20,        
            num_beams=4,          
            length_penalty=2.0,   
            early_stopping=True,
            no_repeat_ngram_size=3
            
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return jsonify({"impression": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)