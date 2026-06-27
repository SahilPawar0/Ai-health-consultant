from flask import Blueprint, request, jsonify
import pickle

disease_bp = Blueprint("disease", __name__)

# Load the newly trained models!
model = pickle.load(open("model/disease_model.pkl", "rb"))
symptom_list = pickle.load(open("model/symptoms.pkl", "rb"))

DISEASE_INFO = {
    "Fungal infection": "A common skin condition caused by fungi, resulting in redness, itching, and flaking of the skin.",
    "Allergy": "An immune system reaction to a foreign substance, like pollen or certain foods, causing sneezing, rashes, or itching.",
    "GERD": "Gastroesophageal reflux disease is a digestive disorder where stomach acid frequently flows back into the esophagus.",
    "Chronic cholestasis": "A condition where the flow of bile from the liver is reduced or blocked, often causing intense itching and jaundice.",
    "Drug Reaction": "An adverse physical reaction to a medication, often presenting as a severe skin rash, fever, or breathing difficulty.",
    "Peptic ulcer diseae": "Painful sores that develop on the inner lining of the stomach, lower esophagus, or small intestine.",
    "AIDS": "Acquired immunodeficiency syndrome, a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV).",
    "Diabetes ": "A chronic condition that affects how your body turns food into energy, resulting in elevated blood sugar levels.",
    "Gastroenteritis": "An intestinal infection marked by diarrhea, cramps, nausea, vomiting, and fever (often called stomach flu).",
    "Bronchial Asthma": "A condition in which a person's airways become inflamed, narrow and swell, producing extra mucus and making it difficult to breathe.",
    "Hypertension ": "High blood pressure; a condition in which the long-term force of the blood against your artery walls is dangerously high.",
    "Migraine": "A neurological condition that can cause multiple symptoms, frequently characterized by intense, debilitating headaches.",
    "Cervical spondylosis": "Age-related wear and tear affecting the spinal disks in your neck.",
    "Paralysis (brain hemorrhage)": "Loss of muscle function in part of your body, in this case, caused by bleeding inside the brain.",
    "Jaundice": "A condition in which the skin, whites of the eyes, and mucous membranes turn yellow due to a high level of bilirubin.",
    "Malaria": "A disease caused by a plasmodium parasite, transmitted by the bite of infected mosquitoes, causing high fever and chills.",
    "Chicken pox": "A highly contagious viral infection causing an itchy, blister-like rash on the skin.",
    "Dengue": "A mosquito-borne viral disease occurring in tropical and subtropical areas, causing high fever and severe joint pain.",
    "Typhoid": "A bacterial infection that can lead to a high fever, diarrhea, and vomiting, usually spread through contaminated food or water.",
    "hepatitis A": "A highly contagious liver infection caused by the hepatitis A virus, preventable by vaccine.",
    "Hepatitis B": "A serious liver infection caused by the hepatitis B virus that can be easily prevented by a vaccine.",
    "Hepatitis C": "An infection caused by a virus that attacks the liver and leads to inflammation.",
    "Hepatitis D": "A severe viral liver disease that only occurs in people infected with the hepatitis B virus.",
    "Hepatitis E": "A liver disease caused by the hepatitis E virus, mainly transmitted through contaminated drinking water.",
    "Alcoholic hepatitis": "Inflammation of the liver caused by drinking too much alcohol over a long period.",
    "Tuberculosis": "A potentially serious infectious bacterial disease that mainly affects the lungs.",
    "Common Cold": "A common viral infection of the nose and throat that usually resolves within a week.",
    "Pneumonia": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus.",
    "Dimorphic hemmorhoids(piles)": "Swollen and inflamed veins in the rectum and anus that cause discomfort and bleeding.",
    "Heart attack": "A medical emergency where a blood clot blocks blood flow to the heart, potentially causing tissue damage.",
    "Varicose veins": "Gnarled, enlarged veins, most commonly appearing in the legs and feet.",
    "Hypothyroidism": "A condition in which the thyroid gland doesn't produce enough of certain crucial hormones.",
    "Hyperthyroidism": "The overproduction of a hormone by the butterfly-shaped gland in the neck (thyroid).",
    "Hypoglycemia": "A condition in which your blood sugar (glucose) level is lower than normal.",
    "Osteoarthristis": "The most common form of arthritis, occurring when the protective cartilage that cushions the ends of your bones wears down.",
    "Arthritis": "The swelling and tenderness of one or more of your joints, causing joint pain and stiffness.",
    "(vertigo) Paroymsal  Positional Vertigo": "Episodes of dizziness and a sensation of spinning with certain head movements.",
    "Acne": "A skin condition that occurs when your hair follicles become plugged with oil and dead skin cells.",
    "Urinary tract infection": "An infection in any part of the urinary system, the kidneys, bladder, or urethra.",
    "Psoriasis": "A condition in which skin cells build up and form scales and itchy, dry patches.",
    "Impetigo": "A highly contagious skin infection that causes red sores on the face."
}

@disease_bp.route("/predict-disease", methods=["POST"])
def predict_disease():
    try:
        data = request.json
        symptoms = data.get("symptoms", [])

        if not symptoms:
            return jsonify({"error": "No symptoms provided"})

        # Initialize the array based on the NEW symptom list
        input_vector = [0] * len(symptom_list)

        # Map symptoms safely
        for symptom in symptoms:
            clean_input = symptom.lower().strip().replace(" ", "_")
            if clean_input in symptom_list:
                idx = symptom_list.index(clean_input)
                input_vector[idx] = 1

        if sum(input_vector) == 0:
            return jsonify({"error": "Symptoms were not recognized by the AI model."})

        # Generate Real ML Predictions!
        prediction_probs = model.predict_proba([input_vector])[0]
        disease_classes = model.classes_

        max_idx = prediction_probs.argmax()
        max_prob = prediction_probs[max_idx]
        predicted_disease = disease_classes[max_idx]

        explanation = DISEASE_INFO.get(
            predicted_disease, 
            "A specific description for this condition is currently being updated in our medical database."
        )

        most_likely = {
            "disease": predicted_disease,
            "confidence": round(max_prob * 100, 2),
            "description": explanation
        }

        # Gather secondary possibilities
        other_possibilities = [
            {
                "disease": disease_classes[i],
                "confidence": round(prob * 100, 2)
            }
            for i, prob in enumerate(prediction_probs)
            if disease_classes[i] != predicted_disease and prob > 0.05
        ]

        other_possibilities = sorted(
            other_possibilities,
            key=lambda x: x["confidence"],
            reverse=True
        )[:3]

        result = {
            "most_likely": most_likely,
            "others": other_possibilities
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})