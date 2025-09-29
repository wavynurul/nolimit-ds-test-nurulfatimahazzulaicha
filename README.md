# Nolimit DS Test - Nurul Fatimah Azzulaicha 🚀

**Customer Support Ticket Prioritization App**  

This Streamlit app predicts the priority of customer support tickets (High / Medium / Low) using DistilBERT embeddings and Logistic Regression. Input tickets manually or upload a CSV for batch predictions.

---

## Quickstart 🚀

Get the app running in under 5 minutes:

1. **Clone the GitHub repository:**

```bash
git clone https://github.com/your-username/nolimit-ds-test-nurulfatimahazzulaicha.git
cd nolimit-ds-test-nurulfatimahazzulaicha
````

2. **Download model files from Hugging Face:**

Go to [Hugging Face Space](https://huggingface.co/spaces/wavynurul09/nolimit-ds-test-nurulfatimahazzulaicha/tree/main/src) and download:

* `ticket_priority_clf.joblib`
* `label_encoder.joblib`
* `distilbert_tokenizer.joblib`
* `distilbert_model.joblib`

Place them in the `src` folder.

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app:**

```bash
streamlit run src/streamlit_app.py
```

5. **Start predicting tickets!**
   Type manually or upload a CSV with `Ticket Subject` and `Ticket Description`, then click **Predict** or **Predict All Tickets**.

---

## Features

* Predict ticket priority based on subject and description.
* Supports single ticket input and CSV batch uploads.
* Download predictions as a CSV file.
* Realistic business use case simulation.

---

## Flowchart

Below is the flowchart of the ticket prioritization system:
![Uploading flowchart_system.drawio.png…]()


**Flow Overview:**

```
[User Input] --> [Combine Subject + Description] --> [DistilBERT Embedding] --> [Logistic Regression Prediction] --> [Output Result & Download CSV]
```

* **User Input**: Enter ticket manually or upload CSV with `Ticket Subject` and `Ticket Description`.
* **Combine Subject + Description**: Merge the two columns into one text field.
* **DistilBERT Embedding**: Convert text to embeddings for the model.
* **Logistic Regression Prediction**: Predict ticket priority (High / Medium / Low).
* **Output**: Show predicted priority in app and allow CSV download for batch input.

---

## How to Use (Detailed)

1. Clone the GitHub repository or open in Hugging Face Spaces.
2. Download the model files from Hugging Face [here](https://huggingface.co/spaces/wavynurul09/nolimit-ds-test-nurulfatimahazzulaicha/tree/main/src) and place them in the `src` folder.
3. Run the Streamlit app:

```bash
streamlit run src/streamlit_app.py
```

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
└─ src/                   # Folder for training scripts / model placeholders
   ├─ train_model.ipynb   # Example training notebook (optional)
   ├─ ticket_priority_clf.joblib   # Downloaded from Hugging Face
   ├─ label_encoder.joblib         # Downloaded from Hugging Face
   ├─ distilbert_tokenizer.joblib  # Downloaded from Hugging Face
   └─ distilbert_model.joblib      # Downloaded from Hugging Face
```

> ⚠️ Note: Model files are large and cannot be stored on GitHub. Use the Hugging Face link above to access them.

---

## License

MIT License

---

## References

* [Streamlit Documentation](https://docs.streamlit.io/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
* [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

```

