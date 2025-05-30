# 🕵️‍♂️ Fraud Detection System using Machine Learning

Detect fraudulent credit card transactions with high accuracy using SMOTE, Random Forest, and a custom CLI interface.

---

## 📦 Repository

**GitHub Repo:** [fraud-detection-ml-pipeline](https://github.com/ahsankhizar5/fraud-detection-ml-pipeline)

---

## ✨ Features

- 💳 Real-world dataset: Credit Card Transactions (284K+ entries)
- ⚖️ SMOTE oversampling to handle class imbalance
- 🌲 Random Forest model for fraud classification
- 📈 Performance metrics: Accuracy, Precision, Recall, F1-score
- 🧪 CLI interface for testing your own transactions
- ✅ Clean, script-based structure (no notebooks)

---

## Dataset

This project uses the **Credit Card Fraud Detection** dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

To use it:

1. Download the dataset manually from the link above.
2. Extract the `creditcard.csv` file.
3. Place it in the project root directory.

> Note: The dataset is not included in this repository due to GitHub's file size limitations.

## 🚀 Getting Started

Clone the repo and run the script locally.

```bash
# Clone the repo
git clone https://github.com/ahsankhizar5/fraud-detection-ml-pipeline.git
cd fraud-detection-ml-pipeline

# Install dependencies (if not already installed)
pip install pandas scikit-learn imbalanced-learn

# Run the detection script
python fraud_detection.py
````

---

## 🛠️ Tech Stack

- **Python 3**
- **Pandas**
- **Scikit-learn**
- **imbalanced-learn (SMOTE)**
- **RandomForestClassifier**

## 📊 Evaluation Metrics

| Metric    | Score                                       |
| --------- | ------------------------------------------- |
| Accuracy  | 0.9999                                      |
| Precision | 1.00                                        |
| Recall    | 1.00                                        |
| F1-Score  | 1.00                                        |
| ROC-AUC   | ✅ (included in model logic, not shown here) |

---

## 🧾 Project Structure

```
fraud-detection-ml-pipeline/
│
├── creditcard.csv           # Dataset file
├── fraud_detection.py       # Core training + testing pipeline
└── README.md                # You're here!
```

---

## 🌟 Support This Project

If this helped you or inspired you, **please consider giving it a star** ⭐ on GitHub:

👉 [Give a Star](https://github.com/ahsankhizar5/fraud-detection-ml-pipeline)

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use, modify, and share with credit.

---

## 📬 Contact

**Ahsan Khizar**
📧 [ahsankhizar5@gmail.com](mailto:ahsankhizar5@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/ahsankhizar)

---

> “Code is not just about solving problems. It’s about building trust, clarity, and real-world impact — one line at a time.”> — *Ahsan Khizar*
