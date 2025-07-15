import streamlit as st

def load_css():
    """Loads custom CSS for Streamlit app styling."""
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
        }
        .title {
            font-size: 3rem;
            font-weight: bold;
            color: #1E3A8A; /* A deep blue */
            text-align: center;
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .header {
            font-size: 2rem;
            font-weight: bold;
            color: #333;
            margin-top: 20px;
            margin-bottom: 10px;
            border-bottom: 2px solid #1E3A8A;
        }
        /* Style Streamlit's default components */
        .st-emotion-cache-1y4p8pa {
            max-width: 95%; /* Make the main container wider */
        }
        </style>
        """,
        unsafe_allow_html=True
    )