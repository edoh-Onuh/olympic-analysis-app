import streamlit as st
import pandas as pd
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from utils import load_css

# --- Page Configuration & Constants ---
st.set_page_config(
    page_title="Olympic Gold Medal Analysis",
    page_icon="🥇",
    layout="wide"
)

DATA_PATH = Path("data/athlete_events.csv")

# --- Load Utilities ---
load_css()

# --- Data Loading & Caching ---
@st.cache_data
def load_data(path):
    """Loads the dataset and performs initial cleaning."""
    try:
        df = pd.read_csv(path)
        # Handle duplicates
        df = df.drop_duplicates()
        # More robust handling of 'Medal' NaNs
        df['Medal'] = df['Medal'].fillna('No Medal')
        return df
    except FileNotFoundError:
        st.error(f"Error: Dataset file not found at '{path}'. Please download it and place it in the 'data' directory.")
        st.stop()

@st.cache_data
def preprocess_for_modeling(df):
    """Prepares data for CatBoost, identifying categorical features without one-hot encoding."""
    # We will predict if an athlete is male or female based on other attributes
    df_model = df.dropna(subset=['Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Sport', 'Year'])
    
    # --- The Fix is Here ---
    # Start with a copy of all the data for X
    X = df_model.copy() 
    # Pop 'Sex' from X. This assigns the column to y AND removes it from X in one step.
    y = X.pop('Sex').apply(lambda x: 1 if x == 'M' else 0) # 1 for Male, 0 for Female

    # CatBoost works best with categorical features identified
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Get indices of categorical features for CatBoost Pool
    categorical_features_indices = [X.columns.get_loc(col) for col in categorical_features]
    
    return X, y, categorical_features_indices

# --- Optuna Objective Function ---
def objective(trial, X_train, y_train, X_test, y_test, cat_features):
    """The objective function for Optuna hyperparameter tuning."""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': False,
        'task_type': 'CPU',
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy',
        'random_seed': 42
    }
    
    model = CatBoostClassifier(**params)
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    eval_pool = Pool(X_test, y_test, cat_features=cat_features)
    
    model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=30, use_best_model=True)
    
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy

# --- Plotting Functions ---
def plot_optimization_history(study):
    fig = optuna.visualization.plot_optimization_history(study)
    st.plotly_chart(fig, use_container_width=True)

def plot_param_importances(study):
    fig = optuna.visualization.plot_param_importances(study)
    st.plotly_chart(fig, use_container_width=True)

def plot_confusion_matrix_func(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Female', 'Male'], y=['Female', 'Male'],
                    color_continuous_scale='Blues')
    fig.update_layout(title_text='Confusion Matrix')
    st.plotly_chart(fig, use_container_width=True)
    
def plot_feature_importance(model, feature_names):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Feature Importance')
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

# --- Main App Logic ---
def main():
    """Main function to run the Streamlit app."""
    
    # Initialize session state
    if "study" not in st.session_state:
        st.session_state.study = None
    if "best_model" not in st.session_state:
        st.session_state.best_model = None

    # --- Load Data ---
    data = load_data(DATA_PATH)
    X, y, cat_features_indices = preprocess_for_modeling(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Sidebar for Controls ---
    st.sidebar.title("Controls")
    st.sidebar.markdown("Use the controls below to tune the model and explore the data.")

    # --- Main Page Layout ---
    st.markdown("<h1 class='title'>🥇 Olympic Gold Medals & Gender Analysis</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Welcome & Data Overview", "Model Hyperparameter Tuning", "Results & Analysis"])

    with tab1:
        st.markdown("<h2 class='header'>Welcome to the Olympic Analysis App</h2>", unsafe_allow_html=True)
        st.write("""
        This application explores the historical Olympic dataset with a focus on gender. 
        - **Data Overview:** Explore a snapshot of the raw dataset.
        - **Model Tuning:** Use hyperparameter optimization (Optuna) to train a CatBoost model that predicts an athlete's gender based on their physical attributes and event details.
        - **Results & Analysis:** Analyze the tuned model's performance and visualize interesting trends in the data, particularly regarding gold medal distribution.
        """)
        with st.expander("Show Dataset Overview"):
            st.dataframe(data.head())
            st.write(f"The dataset has **{data.shape[0]}** rows after cleaning duplicates.")

    with tab2:
        st.markdown("<h2 class='header'>Train a Gender Prediction Model</h2>", unsafe_allow_html=True)
        st.info("Here, we'll use Optuna to find the best hyperparameters for a CatBoost model. The model's goal is to predict an athlete's gender ('M' or 'F').")
        
        n_trials = st.slider('Number of Optimization Trials', min_value=10, max_value=200, value=25, step=5)
        
        if st.button('Start Hyperparameter Tuning', key='start_tuning'):
            with st.spinner('Running optimization... This might take a few minutes.'):
                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, cat_features_indices), n_trials=n_trials)
                st.session_state.study = study
                
                # Train the final best model
                best_params = study.best_params
                best_model = CatBoostClassifier(**best_params, verbose=False, random_seed=42, task_type="CPU")
                train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
                best_model.fit(train_pool)
                st.session_state.best_model = best_model

            st.success('Optimization finished!')
            st.balloons()
            st.info("Tuning complete! Go to the 'Results & Analysis' tab to see the outcome.")

    with tab3:
        st.markdown("<h2 class='header'>Model Performance & Data Insights</h2>", unsafe_allow_html=True)

        if not st.session_state.study:
            st.warning("Please run the hyperparameter tuning on the 'Model Hyperparameter Tuning' tab first.")
        else:
            study = st.session_state.study
            best_model = st.session_state.best_model
            st.subheader("Tuning Results")
            st.write(f"**Best Accuracy:** `{study.best_value:.4f}`")
            st.write("**Best Parameters:**")
            st.json(study.best_params)
            
            col1, col2 = st.columns(2)
            with col1:
                plot_optimization_history(study)
            with col2:
                plot_param_importances(study)

            st.subheader("Model Evaluation on Test Set")
            y_pred = best_model.predict(X_test)
            col3, col4 = st.columns(2)
            with col3:
                plot_confusion_matrix_func(y_test, y_pred)
            with col4:
                plot_feature_importance(best_model, X_train.columns)

        st.markdown("---")
        st.subheader("Exploratory Data Analysis: Gold Medals")
        
        gold_medalists = data[data['Medal'] == 'Gold']
        
        # Select year for analysis
        years = sorted(gold_medalists['Year'].unique())
        selected_year = st.selectbox('Select Year for Gold Medal Analysis', options=years, index=len(years)-1)
        
        year_data = gold_medalists[gold_medalists['Year'] == selected_year]
        
        # Plotting gender distribution for the selected year
        gender_counts = year_data['Sex'].value_counts()
        fig_pie = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index,
                         title=f'Gold Medal Distribution by Gender for {selected_year}', color=gender_counts.index,
                         color_discrete_map={'M': 'blue', 'F': 'pink'})
        st.plotly_chart(fig_pie)

if __name__ == "__main__":
    main()