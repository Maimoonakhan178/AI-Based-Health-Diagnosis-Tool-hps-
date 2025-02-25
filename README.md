Key Features and Functionalities
User Authentication:

Users can register and log in using their credentials.

User details (name, email, age, etc.) are stored in a JSON file (credentials.json).

Disease Prediction:

The app predicts the likelihood of heart disease and PAD based on user inputs.

Uses machine learning models:

Logistic Regression for heart disease prediction.

Random Forest for PAD prediction.

Predictions are made based on user responses to a series of questions.

Chatbot Interaction:

A chatbot interacts with users to collect information about their health.

The chatbot dynamically asks questions based on the disease being predicted (heart disease or PAD).

Validates user inputs and ensures they fall within acceptable ranges.

Report Generation:

Generates a PDF report summarizing the user's health data and prediction results.

The report includes:

Patient information.

Disease prediction probability.

Health recommendations and disclaimers.

Email Integration:

Sends the generated PDF report to the user's email address.

Uses SMTP (Outlook) for email delivery.

Data Visualization:

Provides interactive charts and graphs for analyzing health data.

Includes:

Scatter plots, bar charts, line charts, radar charts, and pie charts.

Visualizations for heart disease and PAD datasets.

Dashboard:

Displays key metrics and insights about heart disease and PAD.

Includes tables, dosage charts, gauges, and risk factors.

Admin Features:

Admins can view patient data and feedback.

Provides a dashboard for monitoring health trends and predictions.

Tech Stack
Backend: Flask (Python)

Frontend: HTML, CSS, JavaScript (with Chart.js for visualizations)

Machine Learning: Scikit-learn (Logistic Regression, Random Forest)

Data Processing: Pandas, NumPy

PDF Generation: ReportLab

Email Integration: SMTP (Outlook)

Data Storage: JSON files (credentials.json), CSV files (dataset.csv, PAD.csv)

How to Run the Application
Install Dependencies:
Ensure you have the required Python libraries installed:

bash
Copy
pip install flask pandas numpy scikit-learn reportlab smtplib
Prepare Datasets:

Ensure dataset.csv and PAD.csv are in the project directory.

These files should contain the necessary columns for training the machine learning models.

Update Email Credentials:

Replace the SMTP credentials in the code with your Outlook email credentials:

python
Copy
SMTP_SERVER = 'smtp.office365.com'
SMTP_PORT = 587
SMTP_USERNAME = 'your_email@outlook.com'
SMTP_PASSWORD = 'your_password'
Run the Application:
Execute the Flask app:

bash
Copy
python app.py
The app will be accessible at http://127.0.0.1:5000.

Key Files and Their Roles
app.py:

Main Flask application file.

Handles routing, user authentication, chatbot interaction, and disease prediction.

credentials.json:

Stores user registration and login details.

dataset.csv:

Contains data for heart disease prediction.

PAD.csv:

Contains data for PAD prediction.

Templates (HTML files):

index_custom.html: Main dashboard.

register.html: User registration and login page.

dashboard.html: Health insights and visualizations.

charts.html, PAD.html: Data visualization pages.

Static Files (CSS, JS):

Used for styling and interactivity.

Potential Improvements
Database Integration:

Replace JSON and CSV storage with a database (e.g., SQLite, PostgreSQL) for scalability.

Enhanced Security:

Hash passwords before storing them.

Use environment variables for sensitive data (e.g., email credentials).

Advanced Machine Learning:

Experiment with more advanced models (e.g., XGBoost, Neural Networks) for better accuracy.

User Interface:

Improve the UI/UX with modern frameworks like Bootstrap or React.

Error Handling:

Add robust error handling for user inputs and external services (e.g., email).

Deployment:

Deploy the app using platforms like Heroku, AWS, or Google Cloud.

Example Use Case
A user registers and logs in.

The chatbot asks a series of health-related questions.

Based on the responses, the app predicts the likelihood of heart disease or PAD.

A PDF report is generated and emailed to the user.

The user can view their health data and predictions on the dashboard.
