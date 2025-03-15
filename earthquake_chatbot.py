import streamlit as st
import google.generativeai as genai
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set up API key for Gemini
os.environ["GOOGLE_API_KEY"] = "My_API"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to check if the question is related to earthquakes
def is_earthquake_related(question):
    keywords = [
        "earthquake", "seismic", "tremor", "richter", "aftershock", "epicenter",
        "fault line", "tectonic", "magnitude", "seismograph", "earthquake safety",
        "earthquake prediction", "plate tectonics", "quake", "ground shaking"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in keywords)

# Function to get a response from Gemini AI
def get_gemini_response(user_input):
    if not is_earthquake_related(user_input):
        return "❌ This chatbot only answers questions related to earthquakes. Please ask something relevant."

    model = genai.GenerativeModel("gemini-1.5-flash")  
    response = model.generate_content(user_input)
    return response.text

# Function to fetch recent earthquake data from USGS API
def get_recent_earthquakes():
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "endtime": datetime.utcnow().strftime("%Y-%m-%d"),
        "minmagnitude": 4.0,
        "limit": 5
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        quakes = data["features"]
        if quakes:
            return [
                f"📍 {quake['properties']['place']} - Magnitude {quake['properties']['mag']} (Time: {datetime.utcfromtimestamp(quake['properties']['time'] / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')})"
                for quake in quakes
            ]
    return ["No significant earthquakes recorded in the past week."]

# Function to provide earthquake safety tips
def get_safety_tips():
    return """
    🏠 **Before an Earthquake:**
    - Secure heavy objects.
    - Identify safe spots like under tables.
    - Prepare an emergency kit.

    🚨 **During an Earthquake:**
    - Drop, Cover, and Hold On!
    - Stay away from windows and heavy furniture.
    - If outside, move to an open area.

    🚑 **After an Earthquake:**
    - Check for injuries and hazards.
    - Avoid damaged buildings.
    - Stay updated via emergency broadcasts.
    """

# Function to analyze uploaded earthquake dataset
def analyze_earthquake_data(df):
    st.subheader("📊 Data Overview")
    st.write(df.head())  # Show first few rows
    
    st.subheader("📈 Data Summary")
    st.write(df.describe())  # Display statistical summary

    st.subheader("📌 Earthquake Magnitude Distribution")
    plt.figure(figsize=(8, 5))
    sns.histplot(df['magnitude'], bins=20, kde=True)
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.title("Distribution of Earthquake Magnitudes")
    st.pyplot(plt)

    st.subheader("🗺️ Earthquake Locations")
    st.map(df[['latitude', 'longitude']])  # Display on map

# Streamlit UI Configuration
st.set_page_config(page_title="Earthquake AI Tool", page_icon="🌍", layout="centered")

st.title("🌍 Earthquake AI Tool")
st.write("Get real-time earthquake updates, safety tips, and AI-powered answers!")

# Sidebar Options
st.sidebar.title("🌍 Earthquake Info")
option = st.sidebar.radio("Select an option:", [
    "🔍 Ask AI", "🌎 Recent Earthquakes", "⚠️ Earthquake Safety",
    "📜 Past Earthquakes", "📞 Emergency Contacts", "📊 Upload & Analyze Data"
])

# User chooses to ask the AI
if option == "🔍 Ask AI":
    user_input = st.text_input("Ask your earthquake-related question here...")
    if st.button("Ask"):
        if user_input:
            with st.spinner("Thinking..."):
                response = get_gemini_response(user_input)
            st.write(response)
        else:
            st.warning("Please enter a question.")

# Display recent earthquake data
elif option == "🌎 Recent Earthquakes":
    st.subheader("🌍 Latest Earthquake Updates")
    earthquakes = get_recent_earthquakes()
    for quake in earthquakes:
        st.write(quake)

# Display earthquake safety tips
elif option == "⚠️ Earthquake Safety":
    st.subheader("🛑 Earthquake Safety Guidelines")
    st.write(get_safety_tips())

# Search for past earthquakes by date
elif option == "📜 Past Earthquakes":
    st.subheader("🔎 Search Past Earthquakes")
    start_date = st.date_input("Start Date", datetime.utcnow() - timedelta(days=30))
    end_date = st.date_input("End Date", datetime.utcnow())

    if st.button("Search"):
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "geojson",
            "starttime": start_date.strftime("%Y-%m-%d"),
            "endtime": end_date.strftime("%Y-%m-%d"),
            "minmagnitude": 4.0,
            "limit": 5
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            quakes = data["features"]
            if quakes:
                for quake in quakes:
                    st.write(f"📍 {quake['properties']['place']} - Magnitude {quake['properties']['mag']} (Time: {datetime.utcfromtimestamp(quake['properties']['time'] / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')})")
            else:
                st.write("No earthquake data found for the given period.")
        else:
            st.write("Error fetching data.")

# Display emergency contacts
elif option == "📞 Emergency Contacts":
    st.subheader("🚨 Emergency Contacts & Resources")
    st.write("""
    - **🇺🇸 USA:** FEMA: 1-800-621-FEMA (3362)
    - **🇮🇳 India:** NDRF: 1078 | NDMA: 011-26701728
    - **🇯🇵 Japan:** Japan Meteorological Agency: 03-3212-8341
    - **🌎 Global:** Red Cross: https://www.redcross.org
    """)

# New feature: Upload & Analyze Data
elif option == "📊 Upload & Analyze Data":
    st.subheader("📊 Upload Your Earthquake Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_columns = {'magnitude', 'latitude', 'longitude'}

        if required_columns.issubset(df.columns):
            analyze_earthquake_data(df)
        else:
            st.error("❌ The uploaded file must contain 'magnitude', 'latitude', and 'longitude' columns.")