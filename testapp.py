from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Absolute base path
BASE_PATH = r"C:\Users\tanag\Downloads\INTELLIGENT-CAREER-GUIDANCE-SYSTEM-main\INTELLIGENT-CAREER-GUIDANCE-SYSTEM-main"

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_PATH, "templates"),
    static_folder=os.path.join(BASE_PATH, "static")
)

@app.route('/')
def career():
    return render_template("hometest.html")


@app.route('/predict', methods=['POST'])
def result():

    result = request.form
    res = result.to_dict(flat=True)

    # 🔥 FIXED FEATURE ORDER (VERY IMPORTANT)
    feature_order = [
        "rate_database_fundamentals",
        "rate_computer_architecture",
        "rate_distributed_computing_systems",
        "rate_cyber_security",
        "rate_networking",
        "rate_development",
        "rate_programming_skills",
        "rate_project_management",
        "rate_computer_forensics_fundamentals",
        "rate_technical_communication",
        "rate_ai_ml",
        "rate_software_engineering",
        "rate_business_analysis",
        "rate_communication_skills",
        "rate_data_science",
        "rate_troubleshooting_skills",
        "rate_graphics_designing"
    ]

    arr = [int(res[feature]) for feature in feature_order]

    data = np.array(arr).reshape(1, -1)

    # Load model
    model_path = os.path.join(BASE_PATH, "careerlast.pkl")
    loaded_model = pickle.load(open(model_path, 'rb'))

    predictions = loaded_model.predict(data)
    pred_proba = loaded_model.predict_proba(data)

    jobs_dict = {
        0: 'AI ML Specialist',
        1: 'API Integration Specialist',
        2: 'Application Support Engineer',
        3: 'Business Analyst',
        4: 'Customer Service Executive',
        5: 'Cyber Security Specialist',
        6: 'Data Scientist',
        7: 'Database Administrator',
        8: 'Graphics Designer',
        9: 'Hardware Engineer',
        10: 'Helpdesk Engineer',
        11: 'Information Security Specialist',
        12: 'Networking Engineer',
        13: 'Project Manager',
        14: 'Software Developer',
        15: 'Software Tester',
        16: 'Technical Writer'
    }
    pred_proba = pred_proba > 0.05
    final_res = {}
    index = 0
    for i, prob in enumerate(pred_proba[0]):
        if prob:
            final_res[index] = jobs_dict[i]
            index += 1
            main_job = predictions[0]
            return render_template(
                "testafter.html",
                final_res=final_res,
                job0=main_job
                )
if __name__ == '__main__':
    app.run(debug=True)