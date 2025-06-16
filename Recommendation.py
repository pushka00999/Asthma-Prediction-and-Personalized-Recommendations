import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from numpy import Array
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import tkinter.messagebox as messagebox

import tensorflow as tf
from tensorflow.keras.models import load_model

custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
model = load_model('Saved Data/model.h5', custom_objects=custom_objects)

def preprocess_and_predict():
    # Collect user inputs
    data_dict = {
        'Tiredness': var_tiredness.get(),
        'Dry-Cough': var_dry_cough.get(),
        'Difficulty-in-Breathing': var_difficulty_breathing.get(),
        'Sore-Throat': var_sore_throat.get(),
        'None_Sympton': var_none_sympton.get(),
        'Pains': var_pains.get(),
        'Nasal-Congestion': var_nasal_congestion.get(),
        'Runny-Nose': var_runny_nose.get(),
        'None_Experiencing': var_none_experiencing.get(),
        'Age_0-9': var_age_0_9.get(),
        'Gender_Female': var_gender_female.get(),
        'Gender_Male': var_gender_male.get()
    }

    # Convert to DataFrame
    data = pd.DataFrame([data_dict])

    # Preprocessing
    # Handle missing values
    imputer = IterativeImputer()
    data_imputed = imputer.fit_transform(data)


    # Feature extraction
    row = pd.Series(data_imputed.flatten())
    percentiles = np.percentile(row, [10, 25, 50, 75, 90])

    feat = pd.Series({
        'mean': np.mean(row),
        'median': np.median(row),
        'mode': row.mode().iloc[0] if not row.mode().empty else np.nan,
        'range': np.ptp(row),
        'iqr': np.subtract(*np.percentile(row, [75, 25])),
        'variance': np.var(row),
        'std_dev': np.std(row),
        'cv': np.std(row) / np.mean(row) if np.mean(row) != 0 else np.nan,
        'percentile_10': percentiles[0],
        'percentile_25': percentiles[1],
        'percentile_50': percentiles[2],
        'percentile_75': percentiles[3],
        'percentile_90': percentiles[4],
        'autocorrelation': row.autocorr()
    })

    # Add lag features
    max_lag = 5  # Example max lag value
    for lag in range(1, max_lag + 1):
        if len(row) > lag:
            feat[f'lag_{lag}'] = row.shift(lag).iloc[lag] if len(row) > lag else 0
        else:
            feat[f'lag_{lag}'] = 0

    feat = np.array(feat)

    feat = np.expand_dims(feat, axis=0)

    feat = np.concatenate([data_imputed, feat], axis=1)

    random_indices = np.random.choice(feat.shape[1], size=21, replace=False)

    # Select the features using the random indices
    selected_features = abs(feat[:, random_indices])

    features = selected_features.reshape(selected_features.shape[0], selected_features.shape[1], 1, 1)

    # Predict using the model
    prediction = Array(model.predict(features)[0][0])

    if prediction > 0.75:
        severity = "Severe"
    elif prediction > 0.5:
        severity = "Moderate"
    else:
        severity = "Mild"

    # Update the result label
    result_label.config(text=f'Prediction: {severity}')

    if severity == "Severe":
        messagebox.showinfo("Severity: Severe",
                            "Dietary Recommendations:\n"
                            "- Focus on anti-inflammatory foods and antioxidants.\n"
                            "- Include omega-3 fatty acids from flaxseeds and walnuts.\n"
                            "- Consult a dietitian for personalized advice.\n\n"
                            "Lifestyle Recommendations:\n"
                            "- Follow your prescribed medication regimen closely.\n"
                            "- Maintain regular medical check-ups.\n"
                            "- Manage stress with techniques like yoga or meditation.\n"
                            "- Control your environment to minimize asthma triggers.\n"
                            )
    elif severity == "Moderate":
        messagebox.showinfo("Severity: Moderate",
                            "Dietary Recommendations:\n"
                            "- Include Vitamin D-rich foods like salmon, fortified milk, and eggs.\n"
                            "- Continue with Vitamin C, E, and A-rich foods.\n\n"
                            "Lifestyle Recommendations:\n"
                            "- Prioritize adequate, quality sleep.\n"
                            "- Engage in manageable physical activity.\n"
                            "- Monitor symptoms and use a peak flow meter if recommended.\n"
                            )
    elif severity == "Mild":
        messagebox.showinfo("Severity: Mild",
                            "Dietary Recommendations:\n"
                            "- Include Vitamin C-rich foods like bell peppers, oranges, strawberries, broccoli.\n"
                            "- Add Vitamin E-rich foods such as sunflower seeds, almonds, avocado.\n"
                            "- Eat Vitamin A and beta-carotene-rich foods like carrots, cantaloupe, sweet potatoes.\n\n"
                            "Lifestyle Recommendations:\n"
                            "- Aim for regular, light to moderate exercise.\n"
                            "- Maintain a healthy weight.\n"
                            "- Avoid known asthma triggers.\n"
                            )
    else:
        messagebox.showinfo("No Asthma",
                            "No asthma symptoms detected based on the current inputs.\n"
                            "Continue monitoring your health and consult a doctor if symptoms develop."
                            )


# Create the main window
root = tk.Tk()
root.title('Health Data Predictor')

# Define variables
var_tiredness = tk.DoubleVar()
var_dry_cough = tk.DoubleVar()
var_difficulty_breathing = tk.DoubleVar()
var_sore_throat = tk.DoubleVar()
var_none_sympton = tk.DoubleVar()
var_pains = tk.DoubleVar()
var_nasal_congestion = tk.DoubleVar()
var_runny_nose = tk.DoubleVar()
var_none_experiencing = tk.DoubleVar()
var_age_0_9 = tk.DoubleVar()
var_gender_female = tk.DoubleVar()
var_gender_male = tk.DoubleVar()

# Create input fields
fields = [
    'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
    'None_Sympton', 'Pains', 'Nasal-Congestion', 'Runny-Nose',
    'None_Experiencing', 'Age_0-9', 'Gender_Female', 'Gender_Male'
]
vars = [
    var_tiredness, var_dry_cough, var_difficulty_breathing, var_sore_throat,
    var_none_sympton, var_pains, var_nasal_congestion, var_runny_nose,
    var_none_experiencing, var_age_0_9, var_gender_female, var_gender_male
]

for i, field in enumerate(fields):
    ttk.Label(root, text=field).grid(row=i, column=0, padx=10, pady=5)
    ttk.Entry(root, textvariable=vars[i]).grid(row=i, column=1, padx=10, pady=5)

# Create a button to preprocess and predict
ttk.Button(root, text='Predict', command=preprocess_and_predict).grid(row=len(fields), column=0, columnspan=2, pady=10)

# Create a label to show the result
result_label = ttk.Label(root, text='')
result_label.grid(row=len(fields) + 1, column=0, columnspan=2, pady=10)

root.mainloop()
