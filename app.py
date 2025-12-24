import os
from flask import Flask, request, jsonify
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Replace with your info
REPO_ID = "CyberCoder225/maira-model" 
FILENAME = "SmolLM2-360M-Instruct.Q4_K_M.gguf"

# This downloads the model from HF to Render's temporary memory
print("Fetching Maira's brain from Hugging Face...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

llm = Llama(model_path=model_path, n_ctx=2048)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "")
    prompt = f"### User: {user_input}\n### Maira:"
    output = llm(prompt, max_tokens=150, stop=["###", "</s>"], echo=False)
    response = output["choices"][0]["text"].strip()
    return jsonify({"maira": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
