# AI-Driven Personalized Health & Diet Consultant

A full-stack healthcare application that bridges the gap between predictive diagnostics and personalized nutritional planning. The system combines a machine learning classification engine with generative artificial intelligence to deliver hyper-personalized health insights and structured diet routines.

---

## 🚀 Key Features
- **Symptom Classification Engine:** Uses a trained Machine Learning model (Random Forest / Decision Tree) to analyze user-inputted symptoms and predict potential conditions.
- **Generative Health Explanations:** Powered by the Google Gemini API to translate raw machine learning outputs into easily digestible, compassionate, and contextual advice.
- **Hyper-Personalized Diet Charts:** Tailors lifestyle, calorie tracking (e.g., 1500-calorie routines), and health recommendations directly to the user's personal biometrics.
- **Modern User Interface:** Built with a clean, highly responsive glassmorphic frontend layout for seamless user interactions.

---

## 🛠️ Tech Stack & Architecture
- **Frontend:** HTML5, CSS3 (Glassmorphism design), JavaScript
- **Backend:** Python, Flask (Web framework)
- **Machine Learning Engine:** Scikit-Learn (Random Forest & Decision Tree models)
- **Generative AI Core:** Google Gemini API

---

## 📦 File Structure
```text
health/
│
├── app.py                     # Main Flask application & API backend
├── index.html                 # Frontend user dashboard & chat interface
├── style.css                  # Glassmorphism visual styles
├── script.js                  # Frontend API handler & UI logic
├── requirements.txt           # List of required Python packages
├── README.md                  # Project documentation (this file)
└── [ML_Model_Files].pkl       # Saved predictive model binaries
⚙️ Local Installation & Setup
1. Clone the Repository
Bash
git clone [https://github.com/your-username/ai-health-consultant.git](https://github.com/your-username/ai-health-consultant.git)
cd ai-health-consultant
2. Set Up Environment Variables
Create a environment variable or configure your key in your system environment:

Bash
# On Windows (Command Prompt)
set GEMINI_API_KEY="your_actual_gemini_api_key"

# On macOS/Linux
export GEMINI_API_KEY="your_actual_gemini_api_key"
3. Install Dependencies
Make sure you have Python installed, then run:

Bash
pip install -r requirements.txt
4. Run the Application
Start the local Flask development server:

Bash
python app.py
