# Student Performance Prediction using Machine Learning

Project Overview
This project predicts student performance (Pass/Fail) using machine learning techniques.  
It analyzes academic data such as study hours, attendance, and previous scores to identify student outcomes.


Objective
- Predict student academic performance (Pass/Fail)
- Compare different machine learning models
- Identify important factors affecting student performance


Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Visual Studio Code


Dataset
The dataset contains student academic and personal information.

Key Features Used:
- Hours_Studied  
- Attendance  
- Previous_Scores  

Target Variable:
- Final_Result (0 = Fail, 1 = Pass)  


Models Used
1. Logistic Regression  
2. Decision Tree  


Methodology
1. Data preprocessing  
2. Feature selection  
3. Train-test split  
4. Model training  
5. Prediction  
6. Evaluation using accuracy  
7. Visualization of results  


Visualizations
- Confusion Matrix  
- Correlation Heatmap  
- Feature Importance  
- Model Comparison Graph  


Project Structure
student-performance-prediction-ml/
│
├── data/
├── graphs/
├── outputs/
├── src/ 
├── README.md
├── requirements.txt


How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the project:
   python src/main.py

Result
Logistic Regression achieved about 83% accuracy in classifying students as pass or fail.

Conclusion
This project demonstrates how machine learning can be used to analyze student data and improve academic outcomes through data-driven insights.
