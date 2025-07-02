import torch
from flask import Flask, render_template, request, redirect, url_for, session, flash
from torchvision import models, transforms
from PIL import Image
import json
import os
import hashlib
import shutil

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key

# Load the pre-trained ResNet50 model
model = models.resnet50(weights='IMAGENET1K_V1')
model.eval()

# Load the class labels for ImageNet
with open('imagenet-simple-labels.json', 'r') as f:
    class_idx = json.load(f)

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper functions for authentication and chat history
def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

def load_chat_history(username):
    chat_file = f"chat_history_{username}.json"
    if os.path.exists(chat_file):
        with open(chat_file, "r") as f:
            return json.load(f)
    return []

def save_chat_history(username, chat_history):
    chat_file = f"chat_history_{username}.json"
    with open(chat_file, "w") as f:
        json.dump(chat_history, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route("/", methods=["GET", "POST"])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    chat_history = load_chat_history(username)

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()

        if user_input:
            bot_response = generate_bot_response(user_input)
            chat_history.append({"user": user_input, "bot": bot_response})

        file = request.files.get('image')
        if file and file.filename:
            image_filename = file.filename
            image_path = os.path.join("static/uploads", image_filename)
            os.makedirs("static/uploads", exist_ok=True)
            file.save(image_path)

            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    max_prob, predicted_class = torch.max(probabilities, 0)

                confidence_threshold = 0.5

                if max_prob.item() > confidence_threshold:
                    label = class_idx[predicted_class.item()]
                    bot_response = f"I see a {label}! Confidence: {max_prob.item():.2f}"
                else:
                    bot_response = "I can't recognize this image with enough confidence."

                chat_history.append({"user": "Uploaded image", "bot": bot_response, "image": f"uploads/{image_filename}"})

            except Exception as e:
                bot_response = f"Error processing the image: {str(e)}"
                chat_history.append({"user": "Uploaded image", "bot": bot_response})

            # Clean up uploaded image
            if os.path.exists(image_path):
                os.remove(image_path)

        save_chat_history(username, chat_history)
        return render_template("index.html", chat_history=chat_history)

    return render_template("index.html", chat_history=chat_history)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        users = load_users()

        if username in users and users[username] == hash_password(password):
            session['username'] = username
            flash("Login successful!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        users = load_users()

        if username not in users:
            users[username] = hash_password(password)
            save_users(users)
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))
        else:
            flash("Username already exists. Try a different one.", "danger")

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop('username', None)
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))

def generate_bot_response(user_input):
    user_input = user_input.lower()
    if "hello" in user_input:
        return "Hello! How can I assist you today?"
    elif "image" in user_input:
        return "Great! Upload an image, and I'll tell you what I see."
    elif "bye" in user_input:
        return "Goodbye! Feel free to come back anytime!"
    elif "hi" in user_input:
        return "Hi! How can I help you today?"
    elif "clear" in user_input:
        username = session['username']
        save_chat_history(username, [])
        return "Chat history cleared!"
    else:
        return "I'm here to help! Try uploading an image or ask me something specific."

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)