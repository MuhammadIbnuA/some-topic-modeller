import streamlit as st
import pandas as pd
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def preprocess_text(text):
    """Preprocesses text for topic modeling."""
    words = text.lower().split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def perform_topic_modeling(df, column_name):
    """Performs topic modeling and returns BERTopic model."""
    df[column_name] = df[column_name].astype(str).apply(preprocess_text)
    reviews = df[column_name].dropna().tolist()

    topic_model = BERTopic(verbose=True)  # Enable verbose for progress updates
    topics, _ = topic_model.fit_transform(reviews)

    return topic_model

st.title("Topic Modeling App")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    column_name = st.selectbox("Select the column for topic modeling:", df.columns)

    if st.button("Run Topic Modeling"):
        topic_model = perform_topic_modeling(df.copy(), column_name)  # Copy DF to avoid modifying the original
        
        # Topic Information
        st.header("Topic Information")
        topic_info = topic_model.get_topic_info()
        st.dataframe(topic_info)
        
        # Topic Visualization
        st.header("Topic Visualization")
        fig = topic_model.visualize_topics()
        st.plotly_chart(fig) 

        # Additional Visualizations
        # Uncomment the visualizations you want to see
        st.header("Topic Hierarchy")
        fig_hier = topic_model.visualize_hierarchy()
        st.plotly_chart(fig_hier)

        st.header("Topic Similarity Matrix")
        fig_sim = topic_model.visualize_heatmap()
        st.plotly_chart(fig_sim)
