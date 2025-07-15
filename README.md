# 🥇 Olympic Gender Analysis & Prediction

This is a Streamlit web application that performs an analysis of the historical Olympic dataset with a focus on gender. The app includes data exploration, interactive visualizations, and a machine learning component to predict an athlete's gender based on their physical and event-related attributes.

The project leverages **CatBoost** for modeling and **Optuna** for efficient hyperparameter tuning.

## ✨ Features

-   **Interactive UI**: A clean, tab-based user interface built with Streamlit.
-   **Efficient Machine Learning**: Uses CatBoost's native handling of categorical features, avoiding inefficient one-hot encoding.
-   **Hyperparameter Tuning**: Integrates Optuna to automatically find the best parameters for the model.
-   **Stateful App**: Uses `st.session_state` to store model tuning results, preventing re-computation and creating a responsive user experience.
-   **Rich Visualizations**: Interactive plots from Plotly for data exploration and model evaluation (e.g., confusion matrix, feature importance).
-   **Robust & Modular Code**: The codebase is organized into logical modules for clarity and maintainability.

---

## 📂 Project Structure

```
olympic-gender-analysis/
├── .gitignore          # Files to be ignored by Git
├── README.md           # This file
├── requirements.txt    # Project dependencies
├── app.py              # Main Streamlit application script
├── utils.py            # Utility functions (e.g., CSS styling)
└── data/
    └── (athlete_events.csv goes here)
```

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd olympic-gender-analysis
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    Download the Olympic History dataset from [Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results).
    Place the `athlete_events.csv` file inside the `data/` directory.

---

## ⚙️ How to Run

From the root directory of the project (`olympic-gender-analysis/`), run the following command in your terminal:

```bash
streamlit run app.py
```

Your web browser should automatically open with the running application.

---

## 🖼️ App Preview

*Navigate the app using the tabs:*
1.  **Welcome & Data Overview**: Get an introduction and view the raw data.
2.  **Model Hyperparameter Tuning**: Start the Optuna optimization process to train the gender prediction model.
3.  **Results & Analysis**: Once tuning is complete, explore the model's performance, feature importance, and interactive charts on gold medal distribution.

![App Screenshot](https://user-images.githubusercontent.com/your-username/your-repo/your-image-link.png) 
*(You can add a screenshot of your app here after uploading it to your GitHub repo)*