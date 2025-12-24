from flask import Flask, request, jsonify
from llama_cpp import Llama

app = Flask(__name__)

# Load Maira - using the exact filename you downloaded
llm = Llama(model_path="SmolLM2-360M-Instruct.Q4_K_M.gguf", n_ctx=2048)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "")
    
    # The format she learned in training
    prompt = f"### User: {user_input}\n### Maira:"
    
    output = llm(prompt, max_tokens=150, stop=["###", "</s>"], echo=False)
    response = output["choices"][0]["text"].strip()
    
    return jsonify({"maira": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)