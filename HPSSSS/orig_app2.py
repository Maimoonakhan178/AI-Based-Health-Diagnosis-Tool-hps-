###############################################IMPORTANT DEPENDENCIES################################################

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, PageTemplate, Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from reportlab.pdfgen import canvas
from sklearn.pipeline import Pipeline
import csv
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import random, re
from reportlab.lib.pagesizes import letter, inch
import json
from datetime import datetime
import os


##################################################INITIALIZING APP#######################################################

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', data=data)
##################################################KNOWLEDGE BASE ACCORDING TO DISEASES#######################################################


questions = {
    "heart_disease": [
        ["To initiate the heart disease category, respond with 'yes.'","Begin with 'yes' if you want to explore questions related to heart disease.","Respond affirmatively with 'yes' to delve into heart disease inquiries.","If you want to discuss heart disease, start by saying 'yes.'","Indicate your interest in heart disease by responding positively with 'yes.'","To commence the heart disease section, reply with 'yes.'","Start the heart disease segment by affirming with 'yes.'","If you're interested in heart disease topics, begin with 'yes.'","Choose 'yes' to kick off the discussion on heart disease.","Respond with 'yes' to initiate questions related to heart disease."],
        ["What is your age?", "How old are you?", "Can I know your age?", "May I ask your age?", "What's your age?", "Could you share your age with me?", "What year were you born?", "How many years have you lived?", "Could you tell me your exact age?", "What age are you currently?"],
        ["What is your gender?", "Could you tell me your gender?", "Are you male or female?", "May I know your gender?", "What gender do you identify with?", "Can you specify your gender?", "Do you identify as male, female, or non-binary?", "What's your gender?", "Could you inform me about your gender identity?", "Which gender do you belong to?"],
        ["What is your resting blood pressure?", "Could you share your resting BP reading?", "What's your blood pressure when resting?", "Can you tell me your current resting blood pressure?", "What is the reading of your blood pressure at rest?", "Could you inform me of your resting BP?", "What's your usual resting blood pressure?", "Can you provide your blood pressure measurement at rest?", "What was your last recorded resting blood pressure?", "Could you specify your resting blood pressure levels?"],
        ["What is your serum cholesterol level?", "Can you provide your cholesterol level?", "What's your current cholesterol reading?", "May I know your serum cholesterol measurement?", "What is the level of your blood cholesterol?", "Could you tell me your cholesterol level?", "What's the count of your serum cholesterol?", "Can you inform me about your cholesterol levels?", "What are your cholesterol numbers like?", "Could you specify your serum cholesterol level?"],
        ["Do you have fasting blood sugar above 120 mg/dL?", "Is your fasting glucose level over 120 mg/dL?", "Does your fasting blood sugar exceed 120 mg/dL?", "Is your blood sugar higher than 120 mg/dL when fasting?", "Could you tell me if your fasting blood glucose is above 120 mg/dL?", "What's your fasting blood sugar reading?", "Do you know if your fasting glucose is more than 120 mg/dL?", "Can you confirm if your fasting blood sugar is over 120 mg/dL?", "Is your blood sugar level high when fasting?", "Do you have elevated fasting blood sugar levels?"],
        ["On a scale from 0 to 2, what are your resting electrocardiographic results (within 0 to 2)?","Provide details about your resting electrocardiographic results, using a scale from 0 to 2.","What is your resting electrocardiographic result, rated on a scale from 0 to 2?","Share the results of your resting electrocardiogram on a scale from 0 to 2.","Specify your resting electrocardiographic results, keeping it within the range of 0 to 2.","On a scale of 0 to 2, rate your current resting electrocardiographic results.", "Can you describe your resting electrocardiographic results using a scale from 0 to 2?","What is the level of your resting electrocardiographic results on a scale from 0 to 2?","Provide insights into your resting electrocardiographic results, within the range of 0 to 2.",  "Rate your resting electrocardiographic results on a scale of 0 to 2."  ],
        ["Share your maximum heart rate achieved during exercise (within the range of 71 to 202).","What's the peak heart rate you've attained during your workout sessions (between 71 and 202)?","Describe the highest heart rate you've reached while exercising (within the range of 71 to 202).","Can you recall the maximum heart rate you achieved during physical activity (71 to 202)?","Specify the top heart rate you experience when working out (within the range of 71 to 202).","Tell me about your max heart rate during your exercise routine (between 71 and 202).","What was your recorded highest heart rate during exercise (within the range of 71 to 202)?","Provide details about the peak heart rate you've observed during physical exertion (71 to 202).","During intense exercise, what's the maximum heart rate you've measured (between 71 and 202)?","Give insights into the peak heart rate achieved during your activity (71 to 202)."],
        ["Do you experience exercise-induced angina?", "Do you have angina during exercise?", "Does physical activity cause you any chest pain or discomfort?", "Do you feel chest pain when exercising?", "Is there angina present during your workouts?", "Can you tell me if you experience chest pain during exercise?", "Does exercise trigger any chest discomfort for you?", "Do you suffer from angina when you're physically active?", "Is there chest pain associated with your exercise routines?", "Do you get angina symptoms during physical exertion?"],
        [ "Specify the ST depression induced by exercise relative to rest (within 0.0 to 6.2).","Describe any changes in ST depression during exercise compared to rest (0.0 to 6.2).","What differences do you notice in ST depression when resting and during exercise (0.0 to 6.2)?", "Provide details about the ST depression from exercise versus rest (within 0.0 to 6.2).","On a scale from 0.0 to 6.2, detail the level of ST depression during exercise compared to rest.","Specify how the ST depression changes from rest to exercise (within 0.0 to 6.2).", "In what ways does the ST depression vary from resting to exercising (0.0 to 6.2)?","Highlight the differences in ST depression when at rest and during exercise (0.0 to 6.2).","Explain the variations in ST depression during physical activity compared to rest (0.0 to 6.2).","Share your observations on changes in ST depression between rest and exercise (0.0 to 6.2)."],
        ["As seen on fluoroscopy, how many major vessels are showing abnormalities (within 0 to 4)?","Provide the count of abnormal vessels in your fluoroscopy results (between 0 and 4).","What did the fluoroscopy reveal about major vessel abnormalities (within 0 to 4)?","Specify the number of abnormal major vessels in your fluoroscopy findings (0 to 4).","Can you indicate the number of affected major vessels in the fluoroscopy (within 0 to 4)?","What's the count of abnormal major vessels in the fluoroscopy results (0 to 4)?","How many major blood vessels are abnormal as per the fluoroscopy (within 0 to 4)?","Share the number of major vessels with abnormalities on fluoroscopy (0 to 4).","What are the fluoroscopy results regarding major vessel abnormalities (within 0 to 4)?","Provide insights into the major vessel abnormalities observed in the fluoroscopy (0 to 4)." ],
        ["On a scale from 0 to 2, rate the intensity of chest pain experienced during exercise.","How would you characterize the type of chest pain you feel during exercise on a 0 to 2 scale?","Describe the severity of chest pain on a scale from 0 to 2 during your exercise routine.","Rate the chest pain you encounter during exercise on a scale of 0 to 2.", "On a 0 to 2 scale, indicate the level of chest pain you typically feel during exercise.", "Provide a rating on the 0 to 2 scale for the chest pain experienced during physical activity.","Characterize the chest pain intensity during exercise using a scale from 0 to 2.","What level of chest pain do you experience during exercise, rated from 0 to 2?","On a scale of 0 to 2, how would you describe the intensity of chest pain during exercise?","Rate the severity of your chest pain during exercise on a scale of 0 to 2."],
        ["On a scale from 0 to 2, describe the slope of the peak exercise ST segment.","Can you explain the peak exercise ST segment's slope, using a scale from 0 to 2?","What's the characteristic of the peak ST segment during exercise, rated from 0 to 2?","Specify the slope of your exercise ST segment on a scale from 0 to 2.","What does the peak exercise ST segment slope look like on a scale of 0 to 2?","Can you describe the ST segment slope at peak exercise, using a scale from 0 to 2?","What are the features of the peak exercise ST segment's slope (0 to 2 scale)?","Illustrate the slope of the ST segment during peak exercise on a scale from 0 to 2.","What is the appearance of the peak exercise ST segment slope (0 to 2 scale)?", "How would you describe the ST segment slope at the height of your exercise (0 to 2)?"],
        ["On a scale from 0 to 3, what is your thalassemia type?","Specify your thalassemia type within the range of 0 to 3.","Can you provide details about your thalassemia type, rated on a scale from 0 to 3?","Share your diagnosed type of thalassemia, keeping it within the range of 0 to 3.","What's your thalassemia type on a scale of 0 to 3?","Provide information about your thalassemia type, ensuring it falls between 0 and 3.","Within the range of 0 to 3, what is your diagnosed type of thalassemia?","Specify your thalassemia type, making sure it's within the range of 0 to 3.", "Can you detail the type of thalassemia you have, which should be between 0 and 3?","On a scale of 0 to 3, indicate your diagnosed type of thalassemia."],
        ["INDEX FOR GENERATING REPORT"],
        ["Wait, Report is generating . . .", "Hold on, compiling your report now...", "Please wait, your report is being prepared...", "Just a moment, generating your report...", "Processing, your report will be ready soon...", "One moment, creating your report...", "Report in progress, please wait...", "Compiling data for your report, hold on...", "Your report is being generated, please be patient...", "Almost there, finalizing your report..."]
  ],
    "PAD": [
    ["To start the health assessment, respond with 'yes.'", "Begin with 'yes' if you want to explore questions related to your health.", "Respond affirmatively with 'yes' to delve into health-related inquiries.", "If you want to discuss your health, start by saying 'yes.'", "Indicate your interest in health by responding positively with 'yes.'", "To commence the health assessment, reply with 'yes.'", "Start the health segment by affirming with 'yes.'", "If you're interested in health topics, begin with 'yes.'", "Choose 'yes' to kick off the discussion on health.", "Respond with 'yes' to initiate questions related to your health."],
   [ "How old are you?","What is your age?","Could you provide your age?","What age range do you fall into?","May I know your current age?","How many years have you been alive?","Tell me your age, please.", "Can you share your age with me?","What's your current age?","Provide me with your age information."],
    ["Do you smoke?", "Are you a smoker?", "Have you been smoking recently?"],
    ["What is your blood pressure reading?", "Have you checked your blood pressure lately?", "Do you know your current blood pressure levels?"],
    ["Have you ever been diagnosed with Peripheral Artery Disease (PAD)?", "Are you aware of any diagnosis of PAD in your medical history?", "Has a healthcare professional ever told you that you have PAD?"],
    ["Do you know your cholesterol levels?", "What is your current cholesterol level?", "Have you had your cholesterol checked recently?"],
    ["Rate your weekly physical activity level on a scale from 1 to 5.", "On a scale of 1 to 5, how active are you during a typical week?", "Please assign a number from 1 to 5 to indicate your weekly physical activity level.", "Indicate your typical weekly physical activity level using a scale from 1 to 5.", "Rate your regular physical activity on a scale of 1 to 5.", "Assign a number between 1 and 5 to describe your usual weekly physical activity.", "On a scale of 1 to 5, how often do you engage in physical activities during the week?", "Use a scale of 1 to 5 to share your average weekly physical activity level.", "Rate your physical activity habits from 1 to 5.", "Assign a number to represent your weekly physical activity level, with 1 being low and 5 being high."],
    ["What is your BMI?", "Could you provide your current BMI?", "Have you calculated your BMI recently?"],
    ["What is your gender?", "Are you male or female?", "Please specify your gender."],
    ["Is there a history of PAD or other cardiovascular conditions in your family?", "Have any of your family members been diagnosed with PAD or related issues?", "Do you know if cardiovascular conditions run in your family?"],
    ["Provide your current blood sugar level.", "What is your latest blood sugar reading?", "Share your current blood sugar measurement.", "Tell me about your blood sugar level.", "What is your present blood sugar reading?", "Provide the numeric value of your blood sugar level.", "Share the latest results of your blood sugar test.", "What is your current blood sugar concentration?", "Tell me your blood sugar level.", "Provide the numeric value of your latest blood sugar check."],
    ["What is your typical diet? Choose from options: 'high-fat', 'balanced', 'low-fiber'.", "Please select your diet type: 'high-fat', 'balanced', 'low-fiber'.", "Indicate your dietary preference: 'high-fat', 'balanced', 'low-fiber'.", "Choose the category that best describes your diet: 'high-fat', 'balanced', 'low-fiber'.", "Specify your diet type: 'high-fat', 'balanced', 'low-fiber'.", "Select the option that matches your typical diet: 'high-fat', 'balanced', 'low-fiber'.", "What best describes your diet? 'high-fat', 'balanced', 'low-fiber'.", "Pick one: 'high-fat', 'balanced', 'low-fiber', to describe your diet.", "Choose from the following options: 'high-fat', 'balanced', 'low-fiber', to indicate your diet.", "Select your diet type from the options: 'high-fat', 'balanced', 'low-fiber'."],
    ["Rate your stress level on a scale from 1 to 10.", "On a scale of 1 to 10, how stressed do you feel?", "Provide a number from 1 to 10 to indicate your current stress level.", "Indicate your stress level using a scale from 1 to 10.", "Rate how stressed you are from 1 to 10.", "Assign a number between 1 and 10 to describe your stress level.", "On a scale of 1 to 10, how would you rate your stress?", "Use a scale of 1 to 10 to share your current stress level.", "Rate your stress from 1 to 10, with 1 being low and 10 being high.", "Assign a number to represent your stress level, with 1 being the least and 10 being the most."],
    ["Would you describe your lifestyle as sedentary?", "Do you have a sedentary lifestyle? ", "Is your lifestyle more on the sedentary side? ", "Do you spend a significant amount of time sitting or being inactive each day? (Yes/No)", "Is your daily routine characterized by a sedentary lifestyle?", "Would you say you have an active lifestyle?", "Do you engage in regular physical activity? ", "Is your lifestyle more on the active side? ", "Do you spend considerable time being active during the day?", "Would you describe yourself as having a sedentary lifestyle?"],
    ["INDEX FOR GENERATING REPORT"],
    ["Wait, Report is generating . . .", "Hold on, compiling your report now...", "Please wait, your report is being prepared...", "Just a moment, generating your report...", "Processing, your report will be ready soon...", "One moment, creating your report...", "Report in progress, please wait...", "Compiling data for your report, hold on...", "Your report is being generated, please be patient...", "Almost there, finalizing your report..."]
]
}

# Initial question index
current_question_index = 0
current_question_list = []
user_responses = []
prediction = ""

#################################################### LOADING DATA #######################################################
def reload():
    #################################################### LOADING DATA #######################################################

    file_path = "C:\\Users\\Lenovo\\Downloads\\dataset.csv"
    df = pd.read_csv(file_path)

    #################################################### PREPROCESSING #######################################################

    df = pd.concat([df, pd.get_dummies(df['cp'], prefix="cp"), pd.get_dummies(df['thal'], prefix="thal"), pd.get_dummies(df['slope'], prefix="slope")], axis=1)
    df = df.drop(['cp', 'thal', 'slope'], axis=1)

    #################################################### Split the data into training and testing sets #############################################################

    y_data = df.target.values
    x_data = df.drop(['target'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    ################################################### Scale the input data ###########################################################################

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)


    ###################################################  Train a Logistic Regression model ###################################################
    
    lr = LogisticRegression(C=0.01, class_weight='balanced', max_iter=100, multi_class='ovr', penalty='l2', solver='newton-cg', tol=0.0001, warm_start=True)
    lr.fit(x_train_scaled, y_train)
    return scaler, lr, x_test_scaled, y_test
    #################################################### LOADING DATA #######################################################
def reload_pad():
    # Assuming file_path_df1 is the correct file path for your CSV file
    file_path_df1 = "PAD.csv"
    df1 = pd.read_csv(file_path_df1)

    #################################################### PREPROCESSING #######################################################
    # Assuming df1 is your DataFrame
    df1['Age'] = df1['Age'].astype(int)
    df1['BloodPressure'] = df1['BloodPressure'].astype(int)
    df1['CholesterolLevel'] = df1['CholesterolLevel'].astype(int)
    df1['PhysicalActivity'] = df1['PhysicalActivity'].astype(int)
    df1['BMI'] = df1['BMI'].astype(int)
    df1['BloodSugarLevel'] = df1['BloodSugarLevel'].astype(int)
    df1['StressLevel'] = df1['StressLevel'].astype(int)

    # Now, the specified columns have their data types changed to int64
    smoking_mapping = {'Yes': 1, 'No': 0}
    # Use the map function to apply the mapping
    df1['Smoking'] = df1['Smoking'].map(smoking_mapping)

    gender_mapping = {'Male': 1, 'Female': 0}
    # Use the map function to apply the mapping
    df1['Gender'] = df1['Gender'].map(gender_mapping)

    diet_mapping = {'Low-Fiber': 0, 'Balanced': 1, 'High-Fat': 2}
    df1['Diet'] = df1['Diet'].map(diet_mapping)
    # Assuming df is your DataFrame

    # Select only numeric columns for scaling
    numeric_columns = df1.select_dtypes(include='number').columns

    # Create a StandardScaler object
    scaler2 = StandardScaler()

    # Fit and transform the selected numeric columns
    df1[numeric_columns] = scaler2.fit_transform(df1[numeric_columns])

    # Split the data into training and testing sets (e.g., 80% training, 20% testing)
    X_train, X_test, y_train, y_test_rf = train_test_split(df1.drop('DiagnosedPAD', axis=1), df1['DiagnosedPAD'], test_size=0.2, random_state=42)

    # Now, X_train and y_train are the training features and target, and X_test and y_test are the testing features and target

    # Create a Random Forest classifier model
    rf_model = RandomForestClassifier(random_state=42)

    # Train the model on the training data
    rf_model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred_rf = rf_model.predict(X_test)
    return scaler2, rf_model, y_pred_rf


################################################ HOME PAGE ######################################################################
@app.route("/home")
def home():
    with open('credentials.json', 'r') as file:
        data = json.load(file)
    latest_user_info = get_latest_user(data)
    user_email = latest_user_info['Email']
    user_name = latest_user_info['Name']
    return render_template("index_custom.html", user_email = user_email, user_name = user_name)

################################################ COMMUNICATING TO INDEX_CUSTOM.HTML PAGE ######################################################################

@app.route("/get_response", methods=["POST"])
def get_response():
    global current_question_index, user_responses, current_question_list

    user_response = request.form["user_message"]
    
    ######################CHECK SYMPTOMS########################
    
    if not current_question_list and current_question_index == 0:
        if "sick" in user_response.lower():
                        
                    ###################### UPDATING JSON WITH RELATABLE DISEASE ########################

            current_question_list = questions["heart_disease"]
            with open('credentials.json', 'r') as file:
                data = json.load(file)
            latest_user_info = get_latest_user(data)
            latest_user_info['Disease to be predicted'] = "Heart Disease"

            # Save the updated data back to the file
            with open('credentials.json', 'w') as file:
                json.dump(data, file, indent=2)

                
        ######################CHECK SYMPTOMS########################
                
        if "health" in user_response.lower():
            current_question_list = questions["PAD"]
            
                    ###################### UPDATING JSON WITH RELATABLE DISEASE ########################

            with open('credentials.json', 'r') as file:
                data = json.load(file)

            latest_user_info = get_latest_user(data)
            latest_user_info['Disease to be predicted'] = "PAD Disease"

            # Save the updated data back to the file
            with open('credentials.json', 'w') as file:
                json.dump(data, file, indent=2)


    #########################################Check if the current question is in the prediction category##############################################
    
    if (current_question_list == questions["heart_disease"] and current_question_index == 0 and "Yes" in user_response.lower()) or (current_question_list == questions["PAD"] and current_question_index == 0 and "Yes" in user_response.lower()):
        current_question = random.choice(current_question_list[current_question_index])
    else:
        current_question = random.choice(current_question_list[current_question_index])

    if (current_question_list == questions["heart_disease"] and current_question_index == 0 and "no" in user_response.lower()) or (current_question_list == questions["PAD"] and current_question_index == 0 and "Yes" in user_response.lower()):
        return {"bot_response": "If there's any assistance you need, feel free to ask!", "user_responses": user_responses}

    bot_response, _ = custom_chatbot(current_question, user_response, current_question_index)
    current_question_index = _

    user_responses.append({"question": current_question, "response": user_response})

    current_question_index += 1

    if current_question_index >= len(current_question_list):
        current_question_index = 0

    return {"bot_response": bot_response, "user_responses": user_responses}

##################################################### CHATBOT ##################################################3
def custom_chatbot(question, user_response, current_question_index):
    global user_responses
    user_responses.append(user_response)
    extracted_digits = re.findall(r'\b\d+\b', user_response)
    if current_question_list == questions["heart_disease"]:
        # Age
        if (
            current_question_list == questions["heart_disease"]
            and current_question_index == 2
            and (int(extracted_digits[0]) < 15 or int(extracted_digits[0]) > 80)
        ):
            current_question_index -= 1
            return "Age should be between 15 and 80. Please provide correct details.", current_question_index

        # #################### MALE FEMALE 3 ######################################
        
        # Add relevant conditions for Male/Female type questions
        
        #####################################################################

        # "Type 0", "Type 1", "Type 2", "Type 3"
        if (
            current_question_list == questions["heart_disease"]
            and current_question_index == 12
            and (int(extracted_digits[0]) <= 0 or int(extracted_digits[0]) >= 3)
        ):
            current_question_index -= 1
            return "Chest Pain Type should be between 0 and 2. Please provide correct details.", current_question_index

        # Resting Blood Pressure
        if (
            current_question_list == questions["heart_disease"]
            and current_question_index == 4
            and (int(extracted_digits[0]) <= 94 or int(extracted_digits[0]) >= 200)
        ):
            current_question_index -= 1
            return "Resting Blood Pressure should be between 94 and 200. Please provide correct details.", current_question_index

        # "126, 564, 246"
        # Cholesterol
        if (
            current_question_list == questions["heart_disease"]
            and current_question_index == 5
            and (int(extracted_digits[0]) <= 126 or int(extracted_digits[0]) >= 564)
        ):
            current_question_index -= 1
            return "Cholesterol should be between 126 and 564. Please provide correct details.", current_question_index

        ########################## MG/DL #######################################
        
        # Add relevant conditions for MG/DL questions
        
        #########################################################################

        # 0, 1, 2
        # Resting Electrocardiographic Results
        if (
            current_question_list == questions["heart_disease"]
            and current_question_index == 7
            and (int(extracted_digits[0]) <= 0 or int(extracted_digits[0]) >= 2)
        ):
            current_question_index -= 1
            return "Resting Electrocardiographic Results should be between 0 and 2. Please provide correct details.", current_question_index

        # 71, 202, 150
        # Maximum Heart Rate Achieved
        if (
            current_question_list == questions["heart_disease"]
            and current_question_index == 8
            and (int(extracted_digits[0]) <= 71 or int(extracted_digits[0]) >= 202)
        ):
            current_question_index -= 1
            return "Maximum Heart Rate Achieved should be between 71 and 202. Please provide correct details.", current_question_index

        ########################## EXERCISE INDUCED ################################
        
        # Add relevant conditions for Exercise Induced questions
        
        ############################################################################

        # 0.0, 6.2, 1.0
        # ST Depression Induced by Exercise
        if (
            current_question_list == questions["heart_disease"]
            and current_question_index == 10
            and (int(extracted_digits[0]) <= 0.0 or int(extracted_digits[0]) >= 6.2)
        ):
            current_question_index -= 1
            return "ST Depression Induced by Exercise should be between 0.0 and 6.2. Please provide correct details.", current_question_index

        # 0, 4, 1
        # Number of Major Vessels Colored by Fluoroscopy
        if (
            current_question_list == questions["heart_disease"]
            and current_question_index == 11
            and (int(extracted_digits[0]) <= 0 or int(extracted_digits[0]) >= 4)
        ):
            current_question_index -= 1
            return "Number of Major Vessels Colored by Fluoroscopy should be between 0 and 4. Please provide correct details.", current_question_index

        # "Slope 0", "Slope 1", "Slope 2"
        if (
            current_question_list == questions["heart_disease"]
            and current_question_index == 13
            and (int(extracted_digits[0]) <= 0 or int(extracted_digits[0]) >= 2)
        ):
            current_question_index -= 1
            return "Slope Type should be between 0 and 2. Please provide correct details.", current_question_index

        # # "Thal 0", "Thal 1", "Thal 2", "Thal 3"
        if (
            current_question_list == questions["heart_disease"]
            and current_question_index == 14
            and (int(extracted_digits[0]) <= 0 or int(extracted_digits[0]) >= 3)
        ):
            current_question_index -= 1
            return "Thalasemia Type should be between 0 and 3. Please provide correct details.", current_question_index

        # 0.0, 6.2, 1.0
        if not user_response:
            return "Please provide an answer to the question.", current_question_index
        
        # this is for checking if the list is left with 2 index means its time to predict disease 
        if current_question_index == len(current_question_list) - 2:
            
            prediction_result = predict_disease(user_responses,current_question_list)
            return prediction_result,current_question_index
        else:
            return question, current_question_index
    
    if current_question_list == questions["PAD"]:
        
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 2
            and (int(extracted_digits[0]) < 15 or int(extracted_digits[0]) > 80)
        ):
            current_question_index -= 1
            return "Age should be between 15 and 80. Please provide correct details.", current_question_index

        # Smoking
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 3
            and user_response.lower() not in ["yes", "no"]
        ):
            current_question_index -= 1
            return "Please provide a valid response (Yes/No) for Smoking.", current_question_index

        # Blood Pressure
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 4
            and (int(extracted_digits[0]) < 80 or int(extracted_digits[0]) > 200)
        ):
            current_question_index -= 1
            return "Blood Pressure should be between 80 and 200. Please provide correct details.", current_question_index

        # Diagnosed PAD
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 5
            and user_response.lower() not in ["yes", "no"]
        ):
            current_question_index -= 1
            return "Please provide a valid response (Yes/No) for Diagnosed PAD.", current_question_index

        # Cholesterol Level
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 6
            and (int(extracted_digits[0]) < 100 or int(extracted_digits[0]) > 300)
        ):
            current_question_index -= 1
            return "Cholesterol Level should be between 100 and 300. Please provide correct details.", current_question_index

        # Physical Activity
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 7
            and (float(extracted_digits[0]) < 1.0 or float(extracted_digits[0]) > 5.0)
        ):
            current_question_index -= 1
            return "Physical Activity should be between 1.0 and 5.0. Please provide correct details.", current_question_index

        # BMI
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 8
            and (float(extracted_digits[0]) < 15.0 or float(extracted_digits[0]) > 40.0)
        ):
            current_question_index -= 1
            return "BMI should be between 15.0 and 40.0. Please provide correct details.", current_question_index

        # Gender
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 9
            and user_response.lower() not in ["male", "female"]
        ):
            current_question_index -= 1
            return "Please provide a valid response (Male/Female) for Gender.", current_question_index

        # Family History
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 10
            and user_response.lower() not in ["yes", "no"]
        ):
            current_question_index -= 1
            return "Please provide a valid response (Yes/No) for Family History.", current_question_index

        # Blood Sugar Level
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 11
            and (float(extracted_digits[0]) < 70.0 or float(extracted_digits[0]) > 200.0)
        ):
            current_question_index -= 1
            return "Blood Sugar Level should be between 70.0 and 200.0. Please provide correct details.", current_question_index

        # Diet
        valid_diets = ["high-fat", "balanced", "low-fiber"]
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 12
            and user_response.lower() not in valid_diets
        ):
            current_question_index -= 1
            return f"Please provide a valid response ({', '.join(valid_diets)}) for Diet.", current_question_index

        # Stress Level
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 13
            and (float(extracted_digits[0]) < 1.0 or float(extracted_digits[0]) > 10.0)
        ):
            current_question_index -= 1
            return "Stress Level should be between 1.0 and 10.0. Please provide correct details.", current_question_index

        # Sedentary Lifestyle
        if (
            current_question_list == questions["PAD"]
            and current_question_index == 14
            and user_response.lower() not in ["yes", "no"]
        ):
            current_question_index -= 1
            return "Please provide a valid response (Yes/No) for Sedentary Lifestyle.", current_question_index
        if not user_response:
            return "Please provide an answer to the question.", current_question_index
        
        # this is for checking if the list is left with 2 index means its time to predict disease 
        if current_question_index == len(current_question_list) - 2:
            
            prediction_result = predict_disease(user_responses,current_question_list)
            return prediction_result,current_question_index
        else:
            return question, current_question_index



#################################### PREDICITNG DISEASE  ###################################

def predict_disease(user_responses,current_question_list):
    print("USER - REPOSNSES : ", user_responses)
    extracted_value = []
    for i in user_responses:
        input_string = i
        digits = ''.join(filter(str.isdigit, input_string))
        if digits:
            extracted_integer = int(digits)
            extracted_value.append(extracted_integer)
        print("___________________________________________", extracted_value)
            
    if current_question_list == questions["heart_disease"]:
        
        ######################## JUST FOR CHECKING PREDICTION, ACRUALLY VALUES WILL VARIES ON USERS INPUT ######################################

        user_data_df = [63, 1, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]

        #######################################3Scale the input data###################################################
        scaler,lr, x_test_scaled, y_test = reload()
        user_data_scaled = scaler.transform([extracted_value])  # Notice the list inside transform

        #################################### Make a prediction #######################################################
        prediction = lr.predict_proba(user_data_scaled)
        probability_of_positive_class = prediction[0][1]
        y_prediction = lr.predict(x_test_scaled)  # Make predictions on the test set
        accuracy = accuracy_score(y_test, y_prediction)
        global probability
        probability = round(probability_of_positive_class * 100, 2)

        ########################## UPDATING JSON FILE WITH PREDICTION ##########################################

        with open('credentials.json', 'r') as file:
                data = json.load(file)
        latest_user_info = get_latest_user(data)
        latest_user_info['Prediction'] = probability

                # Save the updated data back to the file
        with open('credentials.json', 'w') as file:
            json.dump(data, file, indent=2)
        # Set a threshold (e.g., 0.5) to make a binary prediction
        threshold = 0.5

        if probability_of_positive_class >= threshold:
            
            return f"Our prediction suggests a potential risk of heart disease of {probability:.2f}%. Maintain your well-being with regular check-ups. The model's accuracy on the test set is {round(accuracy,2):.2%}. Say 'Report' to generate a report! "
        else:
            return f"Our prediction suggests a low risk of heart disease {probability:.2f}% . Maintain your well-being with regular check-ups. The model's accuracy on the test set is {round(accuracy,2):.2%}."

    if current_question_list == questions["PAD"]:

    ################### JUST FOR CHECKING PREDICTION, ACTUAL VALUES WILL VARY BASED ON USER INPUT #################

    # Example user input
        user_input = [54, 1, 107, 210, 3, 27, 0, 0, 104, 2, 5, 1]

        # Ensure that the scaler is fitted on the correct training data
         # Replace X_train with the actual training data
        scaler,rf_model, y_test_rf = reload_pad()

        # Make predictions on the scaled user input
        user_input_scaled = scaler.transform([extracted_value])

        # Make predictions using your model (replace 'your_model' with the actual model)
        probability_of_positive_class = rf_model.predict_proba(user_input_scaled)[:, 1]

        # Print the predicted probability of the positive class
        print("Probability of Positive Class:", probability_of_positive_class)

    # Calculate accuracy using the appropriate variables (y_test and y_pred_rf)
        accuracy_pad = accuracy_score(y_test_rf, y_pred_rf)

        # Extract the scalar value from the NumPy array
        probability_scalar = probability_of_positive_class.item()

        # Calculate the probability as a percentage and round it
        probability_pad = round((probability_scalar * 100), 2)

        ########################### UPDATING JSON FILE WITH PREDICTION ##########################################

        with open('credentials.json', 'r') as file:
            data = json.load(file)
        latest_user_info_pad = get_latest_user(data)
        latest_user_info_pad['Prediction'] = probability_pad

        # Save the updated data back to the file
        with open('credentials.json', 'w') as file:
            json.dump(data, file, indent=2)

        ########################### SETTING THRESHOLD ##########################################################

        threshold_pad = 0.5

        if probability_of_positive_class >= threshold_pad:
            return f"Our prediction suggests a potential risk of Peripheral Artery Disease of {probability_pad:.2f}%. Consult with your healthcare provider for further evaluation. The model's accuracy on the test set is {round(accuracy_pad,2):.2%}. Say 'YES' to generate a report!"
        else:
            return f"Our prediction suggests a low risk of Peripheral Artery Disease {probability_pad:.2f}% . Continue to maintain your well-being. The model's accuracy on the test set is {round(accuracy_pad,2):.2%}."




     
        
        
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        


##################################### Use a consistent file path for credentials ##############################################


file_path = "credentials.json"  

##################################### Function to load credentials from a file #################################################

def load_credentials():
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            if isinstance(data, dict):
                return data
            else:
                return {'user': {}}
    except (FileNotFoundError, json.JSONDecodeError):
        return {'user': {}}


    
######################################### Function to save credentials to a file ################################################

def save_credentials(credentials):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(credentials, json_file, indent=4)
    except Exception as e:
        print(f"Error saving credentials: {e}")


######################################### LOGIN PAGE ################################################

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Load credentials
        credential = load_credentials()
        print("Credentials:", credential)  # Print the entire credential dictionary

        if username in credential["user"] and credential["user"][username]["Password"] == password:
            print("ok")
            return redirect(url_for('home'))
        else:
            message = 'Incorrect username or password. Please try again.'
            return render_template('register.html', message=message)

    return render_template('register.html')


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["registerUsername"]
        password = request.form["registerPassword"]
        confirm_password = request.form["confirmPassword"]
        email = request.form["registerEmail"]
        full_name = request.form["fullName"]
        age = request.form["age"]

        if password == confirm_password:
            # Load credentials
            credential = load_credentials()
            registration_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            # Create a new dictionary entry for the user
            user_details = {
                "Name": username,
                "Password": password,
                "Email": email,
                "FullName": full_name,
                "Age": age,
                "RegistrationDate": registration_date
            }

            # Add the user details to the credential dictionary
            credential['user'][username] = user_details

            # Save the updated credentials
            save_credentials(credential)

            # Redirect to the login page after successful registration
            return redirect(url_for('login'))
        else:
            error = 'Passwords do not match. Please try again.'
            return render_template('register.html', error=error)

    return render_template('register.html')

@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

########################################### EMAIL PART ####################################################
########################################## CREDIANTIALS #####################################################
SMTP_SERVER = 'smtp.office365.com'
SMTP_PORT = 587
SMTP_USERNAME = 'anasshah444@outlook.com'
SMTP_PASSWORD = '@bukc0230'
from datetime import datetime

########################################## CHECK FOR LATEST PATIENT #####################################################

def get_latest_user(credentials):
    latest_user = max(credentials["user"], key=lambda user: datetime.fromisoformat(credentials["user"][user]["RegistrationDate"]))
    return credentials["user"][latest_user]

########################################## CREATING PDF #####################################################

def create_pdf(user_info):
    # Set up PDF file path
    pdf_file_path = "pdf_file.pdf"

    # Create a PDF document
    pdf = SimpleDocTemplate(pdf_file_path, pagesize=letter)

    # Define header and footer functions
    def header(canvas, doc):
        width, height = letter

        # Blue header background
        canvas.setFillColorRGB(0.2, 0.4, 0.8)
        canvas.rect(0, height - 0.5 * inch, width, 0.5 * inch, fill=True)

        # Header text
        header_text = F"HPS {user_info['Disease to be predicted']} Medical Report ".upper()
        canvas.setFillColorRGB(1, 1, 1)  # Set text color to white
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawCentredString(width / 2.0, height - 0.25 * inch, header_text)


    def footer(canvas, doc):
        width, height = letter
        right_margin = inch * 0.5
        footer_margin = 0.75 * inch

        # Footer background
        canvas.setFillColorRGB(0.2, 0.4, 0.8)
        canvas.rect(0, 0, width, footer_margin, fill=True)

        # Footer text
        footer_text = (
            " MEDIDIAGNOSIS INSIGHTS"
            " Healthcare Analytics Department"
            f" Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Draw footer element
        canvas.saveState()
        canvas.setFillColorRGB(1, 1, 1)  # Set text color to white
        canvas.setFont("Helvetica", 10)
        canvas.drawRightString(width - right_margin, footer_margin / 2, footer_text)
        canvas.restoreState()

    # Create header and footer PageTemplates with Frame instances
    frame_width = pdf.width + 1.5 * inch  # Adjusted width
    frame_height = pdf.height - 1 * inch  # Adjusted space for the footer and content
    header_frame = Frame(inch / 2, frame_height + inch, frame_width, inch, id='header')
    content_frame = Frame(inch / 2, inch, frame_width, frame_height, id='content')
    footer_margin = 0.75 * inch  # Adjusted footer margin

    pdf.addPageTemplates([PageTemplate(id='HeaderFooter', frames=[header_frame, content_frame], onPage=header, onPageEnd=footer)])

    # Set font and styles for the content
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        'CustomStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=8,
        textColor='black'
    )
    story = []

    # Patient Information Section
    story.append(Paragraph("<b>Patient Information:</b>", styles['Heading1']))

    # Create a table for patient information
    patient_table_data = [
        ["Patient Name", user_info['Name']],
        ["Age", user_info['Age']],
        ["Email", user_info['Email']],
        ["Disease", user_info['Disease to be predicted']],
        ["Prediction", user_info['Prediction']],
    ]
    table_style = [
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#3366cc")),  # Blue background
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # Text color for the header row
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
]

    patient_table = Table(patient_table_data, colWidths=[100, 200], style=table_style)

    story.append(patient_table)

    # Sample content
    sample_content = [
        "\n",
        "Taking care of your health is paramount for overall well-being. As a patient, it is crucial to be",
        "proactive in maintaining a healthy lifestyle and adhering to prescribed medical advice. This includes",
        "attending regular check-ups, following medication regimens, and embracing positive habits such as",
        "regular exercise and a balanced diet. Being mindful of stress levels and incorporating sufficient rest into", 
        "your routine also plays a pivotal role in fostering recovery. Remember, your health is a partnership",
         "between you and, your healthcare providers, so communication is key. Stay informed, ask questions, and ",
        "actively participate in decisions regarding your care. By taking these proactive steps, you empower",
        " yourself to contribute to your own well-being and enhance the effectiveness of your treatment plan.",
        "<b>Conclusion:</b>",
        "Based on the reported symptoms and risk factors, as well as the diagnostic recommendations,",
        "the chatbot predicts the likelihood of heart disease. The patient is strongly advised to",
        "follow the recommendations provided and consult with a healthcare professional for further evaluation.",
        "",
        "<b>Disclaimer:</b>",
        "This prediction is based on the information provided by the user and should not replace",
        "professional medical advice. It is crucial for the patient to seek the guidance of a healthcare",
        "professional for a comprehensive evaluation and personalized medical advice."
    ]

    # Add content to the story with reduced spacing after each paragraph
    top_margin = frame_height - 0.5 * inch  # Adjusted top margin for text
    for line in sample_content:
        story.append(Paragraph(line, custom_style))
        top_margin -= 12  # Reduce spacing after each paragraph

    # Build the PDF document
    pdf.build(story)

    return pdf_file_path
########################################## SEND EMAIL FUCNTION #####################################################

def send_email(user_name, user_email):
    subject = 'Health Report'

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    latest_user_info = get_latest_user(data)
    pdf_file = create_pdf(latest_user_info)

    body = f"""Dear {latest_user_info['Name']},\n\nConclusion:
Based on the reported symptoms and risk factors, as well as the diagnostic recommendations, the chatbot predicts the likelihood of heart disease. The patient is strongly advised to follow the recommendations provided and consult with a healthcare professional for further evaluation.

Disclaimer:
This prediction is based on the information provided by the user and should not replace professional medical advice. It is crucial for the patient to seek the guidance of a healthcare professional for a comprehensive evaluation and personalized medical advice.
"""

    msg = MIMEMultipart()
    msg['From'] = SMTP_USERNAME
    msg['To'] = user_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(pdf_file, "rb") as pdf_attachment:
        pdf_attach = MIMEApplication(pdf_attachment.read(), _subtype="pdf")
        pdf_attach.add_header('Content-Disposition', f'attachment; filename={pdf_file}')
        msg.attach(pdf_attach)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_USERNAME, user_email, text)

    print(f"Email sent to {user_email} with subject '{subject}'.")

@app.route('/send_email', methods=['POST'])
def trigger_email():
    data = request.get_json()
    user_name = data.get('user_name')
    user_email = data.get('user_email')

    if user_name and user_email:
        if send_email(user_name, user_email):
            return jsonify({'success': True, 'message': 'Email sent successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to send email'})
    else:
        return jsonify({'success': False, 'message': 'Invalid input data'})



    
    
    
    
    
    
    
    
    
    
    
    

# Read data from the CSV file
data = pd.read_csv('dataset.csv')
data1= pd.read_csv('PAD.csv')





# Route to render the HTML template
@app.route('/dashboard')
def dashboard():
    table_data = []
    table_data1 = []

    # Ensure the file path is correct and accessible
    with open('PAD.csv', mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Append each row as a dictionary
            table_data.append(row)
    # Ensure the file path is correct and accessible
    with open('dataset.csv', mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Append each row as a dictionary
            table_data1.append(row)




    box_data = {
"PERIPHERAL ARTERIAL DISEASE": [
    "Age: Average 60 years",
    "Smoking: 30% smokers",
    "Blood Pressure: Avg. 130 mm Hg",
    "Diagnosed PAD: 20% diagnosed",
    "Cholesterol Level: Avg. 200 mg/dL",
    "Physical Activity: Moderate activity",
    "BMI: Avg. BMI 25 (normal weight)",
    "Gender: Mixed patients",
    "Family History: 40% family history",
    "Blood Sugar Level: Avg. 120 mg/dL",
    "Diet: Diverse dietary patterns",
    "Stress Level: Moderate stress",
    "Sedentary Lifestyle: 25% sedentary"
],




        "Dosage Chart": [
            "Morning: 10mg",
            "Afternoon: 5mg",
            "Evening: 15mg",
            "Night: 20mg",
            "As needed: 10mg",
        ],

        "Gauges": [
            "Blood Pressure: 120/80",
            "Heart Rate: 75 bpm",
            "Stress Level: 98%",
            "Temperature: 98.6°F",
            "Weight: 70kg",
        ],


"HEART DATA OVERVIEW": [
"Avg. Age: 51 years",
"Gender: 60% male, 40% female",
"Chest Pain: Most common type 1",
"Rest BP: Avg. 137 mm Hg",
"Cholesterol: Avg. 255 mg/dL",
"High Fasting BS: 20%",
"Abnormal Rest ECG: 60%",
"Avg. Max HR: 170 bpm",
"Exercise Angina: 80% without",
"Avg. ST Depression: 1.4",
"Upsloping Slope: 70%",
"Avg. Major Vessels: 0.4",
"Thalassemia Type: Most common type 2",
"Heart Disease: Present in 60%"
],




"GOOD SYMPTOMS": [
    "Physical Activity : " "Regular exercise routine",
    "Blood Pressure : " "Stable (e.g., 120/80)",
    "Cholesterol Levels :" "Balanced",
    "Weight : Healthy body weight",
    "Heart Rate" "Normal (e.g., 60-100 bpm)",
    "Chest : No pain or discomfort",
    "Heart Rhythm : No irregular heartbeat",
    "Diet : Healthy, includes fruits and vegetables",
    "Stress Management : Positive stress coping",
    "Check-ups : Regular health check-ups",
    "Smoking : Non-smoker",
    "Family History : No family history of early heart disease"



        ],
        "RISK FACTORS": [
    "Chest Discomfort: Pressure, tightness, or pain", 
    "Shortness of Breath",
    "Unexplained Fatigue",
    "Irregular Heartbeat: Palpitations",
    "Dizziness or Lightheadedness",
    "Excessive Sweating",
    "Nausea or Vomiting",
    "Persistent Cough",
    "Swelling in Legs or Abdomen",
    "Prolonged Pain",
    "Sudden Weakness",
    "Jaw or Throat Pain",
    "Sleep Apnea",
    "High Blood Pressure",
    "Family History of Heart Disease",
    "Diabetes"
],


        "Table": table_data,
        "HEART DISEASE": table_data1,
        
"CRITICAL ZONE": [
    "Avg. Age: 51 years, potential cardiovascular risks.",
    "Gender: 60% male, 40% female, higher risk in males.",
    "Chest Pain: Type 1, a concerning symptom.",
    "Rest BP: Avg. 137 mm Hg, slightly elevated.",
    "Cholesterol: Avg. 255 mg/dL, potential hyperlipidemia.",
    "High Fasting BS: 20%, suggesting diabetes risk.",
    "Abnormal Rest ECG: 60%, indicative of rhythm issues.",
    "Avg. Max HR: 170 bpm, higher than normal at rest.",
    "Exercise Angina: 80% without, potentially dangerous.",
    "Avg. ST Depression: 1.4, indicating myocardial ischemia.",
    "Upsloping Slope: 70%, higher heart disease risk.",
    "Avg. Major Vessels: 0.4, potential vascular issues.",
    "Thalassemia Type: Most common type 2, affects oxygen transport.",
    "Heart Disease: Present in 60%, significant health concern."
],


    }

    return render_template('dashboard.html', box_data=box_data)


@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/about')
def about():
    return render_template('about.html')
    
@app.route('/tables')
def table():
    return render_template('table.html', data=data)

@app.route('/patients')
def patients():
    # Read the content of the credentials.json file
    with open('credentials.json', 'r') as file:
        credentials = json.load(file)

    return render_template('patients.html', credentials=credentials)







# Route to render the HTML template
@app.route('/charts')
def charts():
    # Scatter plot data
    scatter_data = [{'x': row['trestbps'], 'y': row['chol']} for index, row in data.iterrows()]

    # Bar chart data
    bar_labels = list(data.columns)
    bar_labels.remove('target')  # Exclude the 'target' column for demonstration purposes
    bar_chart_data = {
        'labels': bar_labels,
        'datasets': [{
            'label': 'Bar Chart',
            'data': data[bar_labels].mean().tolist(),
            'backgroundColor': 'rgba(70, 190, 192, 0.9)',  # Use the same color as the scatter plot
            'borderColor': 'rgba(75, 192, 192, 1)',  # Border color (optional)
            'borderWidth': 1,  # Border width (optional)
        }],
    }

    # Line chart data for specific columns
    line_labels = ['age', 'thalach']
    line_chart_data = {
        'labels': data.index.tolist(),
        'datasets': [{
            'label': label,
            'data': data[label].tolist(),
            'backgroundColor':'rgba(75, 190, 195, 0.6)',  # Use the same color as the scatter plot
            'borderColor':  'rgba(75, 192, 192, 0.6)',  # Border color (optional)
            'borderWidth': 1,  # Border width (optional)
            'fill': False,  # To create a line chart without fill
        } for label in line_labels],
    }

    # Radar chart data
    exclude_columns = ['target', 'age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca', 'thal']
    radar_labels = [column for column in list(data.columns) if column not in exclude_columns]

    radar_chart_data = {
        'labels': radar_labels,
        'datasets': [{
            'label': 'Radar Chart',
            'data': data[radar_labels].mean().tolist(),
            'backgroundColor': 'rgba(75, 192, 192, 0.6)',  # Use the same color as the scatter plot
            'borderColor': 'rgba(75, 192, 192, 1)',  # Border color (optional)
            'borderWidth': 1,  # Border width (optional)
            'fill': True,  # To fill the radar area
        }],
    }


    # Area chart data
    exclude_columns = ['target', 'age', 'sex','slope','ca',  'trestbps', 'chol', 'thalach', 'exang', 'fbs', 'restecg', 'thal']
    area_labels = [column for column in list(data.columns) if column not in exclude_columns]

    area_chart_data = {
        'labels': data.index.tolist(),
        'datasets': [{
            'label': 'Area Chart',
            'data': data[area_labels].mean().tolist(),
            'backgroundColor': 'rgba(75, 192, 192, 0.6)',  # Use the same color as the scatter plot
            'borderColor': 'rgba(75, 192, 192, 1)',  # Border color (optional)
            'borderWidth': 1,  # Border width (optional)
            'fill': True,  # To fill the area below the line
        }],
    }


  # Pie Chart data for 'cp' column
    cp_labels = data['cp'].value_counts().index.tolist()
    cp_values = data['cp'].value_counts().tolist()
    cp_pie_chart_data = {
        'labels': cp_labels,
        'values': cp_values,
    }


 # Age Histogram data
    age_labels = list(range(min(data['age']), max(data['age']) + 1))
    age_values, _ = np.histogram(data['age'], bins=len(age_labels))
    age_histogram_data = {
        'labels': age_labels,
        'values': age_values.tolist(),
    }
        # Stacked Bar Chart data for 'slope' with 'target' as hue
    slope_labels = data['slope'].value_counts().index.tolist()
    slope_values = []

    for target_value in data['target'].unique():
        slope_values.append(data[data['target'] == target_value]['slope'].value_counts().sort_index().tolist())

    stacked_bar_chart_data = {
        'labels': slope_labels,
        'values': slope_values,
}

# Count plot data for 'target' column
    target_labels = data['target'].value_counts().index.tolist()
    target_values = data['target'].value_counts().tolist()
    target_count_plot_data = {
        'labels': target_labels,
        'values': target_values,
    }


    return render_template('charts.html', scatter_data=scatter_data, bar_chart_data=bar_chart_data, line_chart_data=line_chart_data, radar_chart_data=radar_chart_data, area_chart_data=area_chart_data,cp_pie_chart_data=cp_pie_chart_data,data=data,age_histogram_data=age_histogram_data,stacked_bar_chart_data=stacked_bar_chart_data,target_count_plot_data=target_count_plot_data)
# Route to render the HTML template
@app.route('/PAD')
def PADcharts():
    donut_labels = ['Diagnosed PAD', 'Not Diagnosed PAD']
    donut_values = [int(data1['DiagnosedPAD'].sum()), int(len(data1) - data1['DiagnosedPAD'].sum())]
    donut_chart_data = {
        'labels': donut_labels,
        'values': donut_values,
    }

    # Area Chart data
    area_labels = data1['Smoking'].unique().tolist()
    area_datasets = [{
        'label': 'Smoking Yes' if smoking_status == 'Yes' else 'Smoking No',
        'data': data1[data1['Smoking'] == smoking_status]['BMI'].tolist(),
        'backgroundColor': 'rgba(255, 0, 0, 0.6)' if smoking_status == 'Yes' else 'rgba(75, 192, 192, 0.6)',
        'borderColor': 'white' if smoking_status == 'Yes' else 'rgba(75, 192, 192, 1)',  # White border for 'Smoking Yes'
        'borderWidth': 1,  # Border width (optional)
        'fill': True,  # To fill the area below the line
    } for smoking_status in area_labels]

    area_chart_data = {
        'labels': data1.index.tolist(),
        'datasets': area_datasets,
    }



        # Bar Chart data for Cholesterol Level
    cholesterol_labels = sorted(data1['CholesterolLevel'].unique().astype(int).tolist())[:10]  # Take only the first 10 labels
    cholesterol_values = [len(data1[data1['CholesterolLevel'] == level]) for level in cholesterol_labels]

    cholesterol_chart_data = {
        'labels': cholesterol_labels,
        'datasets': [{
            'label': 'Cholesterol Level',
            'data': cholesterol_values,
            'backgroundColor': ['rgba(255, 0, 0, 0.6)', 'rgba(75, 192, 192, 0.6)', 'rgba(255, 206, 86, 0.6)'],
            'borderColor': ['rgba(255, 0, 0, 1)', 'rgba(75, 192, 192, 1)', 'rgba(255, 206, 86, 1)'],
            'borderWidth': 1,
        }],
    }

  # Bar Chart data for Stress Level
    stress_bar_chart_data = {
        'labels': data1.index.tolist()[:10],  # Take only the first 10 labels
        'datasets': [{
            'label': 'Stress Level',
            'data': data1['StressLevel'].tolist()[:10],  # Take only the first 10 data points
            'backgroundColor': 'rgba(255, 0, 0, 0.6)',
            'borderColor': 'rgba(255, 0, 0, 1)',
            'borderWidth': 1,
        }],
    }

   # Create histogram bins
    hist, bin_edges = np.histogram(data1['Age'], bins=10)

    # Convert ndarray to list
    hist = hist.tolist()

    # Prepare data for the histogram chart
    age_histogram_data = {
        'labels': [int((bin_edges[i] + bin_edges[i + 1]) / 2) for i in range(len(bin_edges) - 1)],
        'datasets': [{
            'label': 'Age Distribution Histogram',
            'data': hist,
            'backgroundColor': 'rgba(75, 192, 192, 0.6)',
            'borderColor': 'rgba(75, 192, 192, 1)',
            'borderWidth': 1,
        }],
    }
        # Line Chart data for Physical Activity
    physical_activity_chart_data = {
        'labels': data1.index.tolist(),
        'datasets': [{
            'label': 'Physical Activity',
            'data': data1['PhysicalActivity'].tolist(),
            'backgroundColor': 'rgba(75, 192, 192, 0.6)',
            'borderColor': 'rgba(75, 192, 192, 0.6)',
            'borderWidth': 1,
            'fill': False,  # Updated to 'false' to remove fill
            'pointRadius': 0,  # Set pointRadius to 0 to remove markers
        }],
    }



        # Bar Chart data for BMI
    bmi_labels = data1.index.tolist()
    bmi_values = data1['BMI'].tolist()

    # Create 15 bins for BMI
    num_bins = 15
    bins = np.linspace(min(bmi_values), max(bmi_values), num_bins + 1)
    binned_data = np.digitize(bmi_values, bins)

    # Calculate the mean BMI for each bin and apply color condition
    bmi_bar_chart_data = {
        'labels': [f'{int(bins[i])}-{int(bins[i + 1])}' for i in range(num_bins)],
        'datasets': [{
            'label': 'BMI Distribution',
            'data': [np.mean(data1['BMI'][binned_data == i]) for i in range(1, num_bins + 1)],
            'backgroundColor': [
                'rgba(255, 0, 0, 0.6)' if np.mean(data1['BMI'][binned_data == i]) > 30 else 'rgba(75, 192, 192, 0.6)'
                for i in range(1, num_bins + 1)
            ],
            
            'borderWidth': 1,
        }],
    }
        # Bar Chart data for Diet
    diet_labels = data1['Diet'].unique().tolist()
    diet_values = [len(data1[data1['Diet'] == diet]) for diet in diet_labels]

    diet_bar_chart_data = {
        'labels': diet_labels,
        'datasets': [{
            'label': 'Diet Distribution',
            'data': diet_values,
            'backgroundColor': ['rgba(255, 0, 0, 0.6)',  'rgba(75, 192, 192, 0.6)' ,'rgba(255, 206, 86, 0.6)'],
            'borderColor': ['rgba(255, 0, 0, 0.6)',  'rgba(75, 192, 192, 0.6)' ,'rgba(255, 206, 86, 0.6)'],
            'borderWidth': 1,
        }],
    }







    

    return render_template('PAD.html', donut_chart_data=donut_chart_data, area_chart_data=area_chart_data,
                        cholesterol_chart_data=cholesterol_chart_data,stress_bar_chart_data=stress_bar_chart_data, age_histogram_data=age_histogram_data, physical_activity_chart_data=physical_activity_chart_data, bmi_bar_chart_data=bmi_bar_chart_data,diet_bar_chart_data =diet_bar_chart_data,data1=data1)






if __name__ == '__main__':
    app.run(debug=True)
