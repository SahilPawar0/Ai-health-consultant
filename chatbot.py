from flask import Blueprint, request, jsonify
from google import genai

chatbot_bp = Blueprint("chatbot", __name__)

# --- PASTE YOUR API KEY HERE ---
API_KEY = "AIzaSyAx8i3MXYncn9sRjQP6CE-HW38y_XdQ_Hc"
client = genai.Client(api_key=API_KEY)

@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")
        
        if not user_message:
            return jsonify({"response": "Please say something."})
            
        # The prompt that forces the AI to act like a health consultant
        prompt = f"""
        You are 'NOVA', a highly intelligent AI Health and Diet Consultant. 
        Answer the following user query accurately and conversationally. 
        If the user asks about symptoms or diseases, provide helpful information but ALWAYS include a brief disclaimer that you are an AI and they should consult a real doctor for medical emergencies.
        Keep your response concise, around 2-3 short paragraphs maximum.
        
        User Query: {user_message}
        """
        
        # Using the brand new SDK and updated 2.5-flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        return jsonify({"response": response.text})
        
    except Exception as e:
        return jsonify({"response": f"API Error: {str(e)}"})