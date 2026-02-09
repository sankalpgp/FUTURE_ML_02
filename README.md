# 🎫 Customer Support Ticket Classification & Prioritization System

An end-to-end **Machine Learning & NLP project** that automatically **classifies customer support tickets**, **assigns priority levels**, and provides a **simple, interactive Gradio web interface** for real-time ticket analysis.

This system helps support teams reduce manual effort, respond faster to urgent issues, and improve customer satisfaction.

---

## 📌 Project Overview

In real-world organizations, customer support teams receive hundreds of tickets daily.  
Manual sorting leads to delays, inefficiency, and missed urgent issues.

This project solves that problem by:
- 🧠 Automatically classifying tickets into categories  
- 🚦 Assigning priorities (High / Medium / Low)  
- 🌐 Providing a **Gradio-based UI** for easy interaction  
- 📊 Displaying prediction confidence  

---

## 🚀 Features

- NLP-based text preprocessing
- Ticket classification using **Logistic Regression**
- Feature extraction using **TF-IDF**
- Rule-based priority assignment
- 📊 Model evaluation with confusion matrix
- 🌐 **Gradio web interface** for real-time predictions
- Simple, clean, and easy-to-understand demo

---

## 🛠️ Tech Stack

### Programming Language
- Python 🐍

### Libraries & Tools
- pandas  
- numpy  
- nltk  
- scikit-learn  
- gradio  
- re (Regular Expressions)

---

## 📂 Dataset

- **File:** `customer_support_tickets.csv`
- **Columns Used:**
  - Ticket Description
  - Ticket Type

---

## 🔄 Project Workflow

1. Load and preprocess dataset  
2. Clean text (lowercase, remove symbols & stopwords)  
3. Convert text to numerical features using **TF-IDF**  
4. Train **Logistic Regression** model  
5. Evaluate performance using accuracy & confusion matrix  
6. Assign ticket priority  
7. Predict ticket category via **Gradio UI**

---

## 🧠 Machine Learning Model

### Logistic Regression
- Efficient for text classification tasks
- Works well with high-dimensional TF-IDF features
- Fast training and interpretable results

---

## 🚦 Priority Assignment Logic

| Priority | Keywords |
|--------|----------|
| High | urgent, failed, error, down |
| Medium | delay, slow, issue |
| Low | general or informational queries |

This ensures urgent tickets are handled first.

---

## 🌐 Gradio Web Interface

The project includes a **Gradio-based interface** that allows users to enter a support ticket and instantly receive:

- ✅ Predicted Ticket Category  
- 🚦 Assigned Priority  
- 📈 Confidence Score  

### Interface Highlights
- Text input box
- JSON output
- Preloaded example tickets
- Runs locally in browser
- Simple and user-friendly design

---

## 🧪 Example Inputs

- *My login is not working and I can't access my account*
- *The payment failed multiple times, this is urgent!*
- *Can you tell me more about the premium features?*

---

## 👨‍💻 Author

- Sankalp Patil
-🎓 Computer Engineering Student
- 🤖 Aspiring Machine Learning Engineer

- 🔗 LinkedIn: https://www.linkedin.com/in/sankalp-patil-sp/
