from flask import Flask
from flask_cors import CORS

# Import your feature blueprints
from disease import disease_bp
from risk import risk_bp
from chatbot import chatbot_bp

app = Flask(__name__)

# Enable CORS so your HTML frontend can talk to the Flask backend
CORS(app)
# Register the routes from your blueprints
app.register_blueprint(disease_bp)
app.register_blueprint(risk_bp)
app.register_blueprint(chatbot_bp)

if __name__ == "__main__":
    # debug=True automatically restarts the server when you save changes
    app.run(debug=True)  
    