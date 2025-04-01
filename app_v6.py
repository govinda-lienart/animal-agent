from flask import Flask, render_template, request
from vector_lookup_v6 import ask_question
import subprocess

app = Flask(__name__)
chat_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history
    if request.method == "POST":
        action = request.form.get("action")
        if action == "ask":
            question = request.form.get("question")
            if question:
                result = ask_question(question)
                chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"]
                })
        elif action == "rebuild":
            subprocess.run(["python", "build_vectorstore_v6b.py"])
            chat_history.clear()
        elif action == "clear":
            chat_history.clear()

    return render_template("index_v6.html", history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)