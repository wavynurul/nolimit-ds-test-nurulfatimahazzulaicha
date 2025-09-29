---
license: mit
title: nolimit-ds-test-nurulfatimahazzulaicha
sdk: streamlit
emoji: 🚀
colorFrom: blue
colorTo: red
short_description: Customer Support Ticket Prioritization App
---
# Nolimit DS Test - Nurul Fatimah Azzulaicha 🚀

**Customer Support Ticket Prioritization App**  

This is a Streamlit app that predicts the priority of customer support tickets (High / Medium / Low) using DistilBERT embeddings and Logistic Regression. The app allows you to input tickets manually or upload a CSV file for batch predictions.

---

## Features

- Predict ticket priority based on subject and description.
- Supports single ticket input and CSV batch uploads.
- Download predictions as a CSV file.
- Realistic business use case simulation.

---

## Flowchart

Below is the flowchart of the ticket prioritization system:

![Flowchart](flowchart.png)

**Flow Overview:**

```

[User Input] --> [Combine Subject + Description] --> [DistilBERT Embedding] --> [Logistic Regression Prediction] --> [Output Result & Download CSV]

````

- **User Input**: Enter ticket manually or upload CSV with `Ticket Subject` and `Ticket Description`.
- **Combine Subject + Description**: Merge the two columns into one text field.
- **DistilBERT Embedding**: Convert text to embeddings for the model.
- **Logistic Regression Prediction**: Predict ticket priority (High / Medium / Low).
- **Output**: Show predicted priority in app and allow CSV download for batch input.

---

## How to Use

1. Clone the repository or open in Hugging Face Spaces.
2. Place your model files (`ticket_priority_clf.joblib`, `label_encoder.joblib`, `distilbert_tokenizer.joblib`, `distilbert_model.joblib`) in the `src` folder.
3. Run the Streamlit app:
   ```bash
   streamlit run src/streamlit_app.py
````

4. Choose input method:

   * **Type manually**: Enter subject and description.
   * **Upload CSV**: Upload CSV with `Ticket Subject` and `Ticket Description` columns.
5. Click "Predict" or "Predict All Tickets" to see results.
6. Download the CSV with predictions if using batch input.

---

## Requirements

* Python 3.8+
* Streamlit
* Pandas
* Joblib
* Torch
* Transformers

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
project_root/
│
├─ app.py                 # Main Streamlit app (outside src)
├─ README.md              # Full README including flowchart reference
├─ requirements.txt       # Dependencies
├─ flowchart.png          # The PNG flowchart used in README
└─ src/                   # Folder for models
   ├─ ticket_priority_clf.joblib
   ├─ label_encoder.joblib
   ├─ distilbert_tokenizer.joblib
   └─ distilbert_model.joblib

```

---

## License

MIT License

---

## References

* [Streamlit Documentation](https://docs.streamlit.io/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
* [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

```