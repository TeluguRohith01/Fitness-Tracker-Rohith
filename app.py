import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time
import warnings

# Configure warnings properly
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set page configuration
st.set_page_config(
    page_title="Personal Fitness Tracker",
    page_icon="🏃‍♂️",
    layout="wide"
)

# Function to load and preprocess data
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)
    
    # Add BMI column
    exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
    exercise_df["BMI"] = round(exercise_df["BMI"], 2)
    
    return exercise_df

# Function to prepare training and test data
@st.cache_data
def prepare_model_data(exercise_df):
    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=42)
    
    # Add BMI column to both training and test sets
    for data in [exercise_train_data, exercise_test_data]:
        data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
        data["BMI"] = round(data["BMI"], 2)
    
    # Prepare the training and testing sets
    exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
    exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)
    
    # Separate features and labels
    X_train = exercise_train_data.drop("Calories", axis=1)
    y_train = exercise_train_data["Calories"]
    
    X_test = exercise_test_data.drop("Calories", axis=1)
    y_test = exercise_test_data["Calories"]
    
    return X_train, y_train, X_test, y_test

# Function to train model
@st.cache_resource
def train_model(X_train, y_train):
    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=42)
    random_reg.fit(X_train, y_train)
    return random_reg

# Function to collect user input and fitness goal

def user_input_features():
    with st.sidebar:
        st.header("User Input Parameters")
        age = st.slider("Age:", 10, 100, 30)
        bmi = st.slider("BMI:", 15, 40, 20)
        duration = st.slider("Duration (min):", 0, 35, 15)
        heart_rate = st.slider("Heart Rate:", 60, 130, 80)
        body_temp = st.slider("Body Temperature (°C):", 36, 42, 38)
        gender_button = st.radio("Gender:", ("Male", "Female"))
        gender = 1 if gender_button == "Male" else 0
        # Fitness goal and experience
        st.header("Workout Personalization")
        goal = st.selectbox("Fitness Goal:", ["Lose Weight", "Build Muscle", "Increase Endurance", "General Fitness"])
        experience = st.selectbox("Experience Level:", ["Beginner", "Intermediate", "Advanced"])
        data_model = {
            "Age": age,
            "BMI": bmi,
            "Duration": duration,
            "Heart_Rate": heart_rate,
            "Body_Temp": body_temp,
            "Gender_male": gender
        }
        features = pd.DataFrame(data_model, index=[0])
    return features, goal, experience

# Function to display progress bar
def show_progress():
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)

# Main application
def get_exercise_library():
    # Exercise: (emoji, video_url)
    return {
        "Bench Press": ("🏋️‍♂️", "https://www.youtube.com/watch?v=gRVjAtPip0Y"),
        "Chest Flys": ("🦋", "https://www.youtube.com/watch?v=eozdVDA78K0"),
        "Push-Ups": ("💪", "https://www.youtube.com/watch?v=_l3ySVKYVJ8"),
        "Pull-Ups": ("🤸‍♂️", "https://www.youtube.com/watch?v=eGo4IYlbE5g"),
        "Lat Pulldown": ("🏋️", "https://www.youtube.com/watch?v=CAwf7n6Luuc"),
        "Bent-Over Row": ("🚣", "https://www.youtube.com/watch?v=vT2GjY_Umpw"),
        "Overhead Press": ("🏋️‍♀️", "https://www.youtube.com/watch?v=qEwKCR5JCog"),
        "Lateral Raise": ("🤲", "https://www.youtube.com/watch?v=kDqklk1ZESo"),
        "Arnold Press": ("🦾", "https://www.youtube.com/watch?v=vj2w851ZHRM"),
        "Bicep Curl": ("💪", "https://www.youtube.com/watch?v=ykJmrZ5v0Oo"),
        "Tricep Dips": ("↕️", "https://www.youtube.com/watch?v=0326dy_-CzM"),
        "Hammer Curl": ("🔨", "https://www.youtube.com/watch?v=zC3nLlEvin4"),
        "Squats": ("🦵", "https://www.youtube.com/watch?v=aclHkVaku9U"),
        "Lunges": ("🚶‍♂️", "https://www.youtube.com/watch?v=QOVaHwm-Q6U"),
        "Leg Press": ("🦿", "https://www.youtube.com/watch?v=IZxyjW7MPJQ"),
        "Deadlifts": ("🏋️", "https://www.youtube.com/watch?v=op9kVnSso6Q"),
        "Planks": ("🧘‍♂️", "https://www.youtube.com/watch?v=pSHjTRCQxIw"),
        "Russian Twists": ("🇷🇺", "https://www.youtube.com/watch?v=wkD8rjkodUI"),
        "Hanging Leg Raises": ("🦵", "https://www.youtube.com/watch?v=JB2oyawG9KI"),
        "Treadmill Running": ("🏃‍♂️", "https://www.youtube.com/watch?v=Qoh3nqpFCF8"),
        "Cycling": ("🚴‍♂️", "https://www.youtube.com/watch?v=1VYze8dRO9w"),
        "Rowing Machine": ("🚣‍♂️", "https://www.youtube.com/watch?v=6ZzFJ1JcB9w"),
        "Jump Rope": ("🤾‍♂️", "https://www.youtube.com/watch?v=1BZM6H8qqQQ"),
        "HIIT Circuits": ("🔥", "https://www.youtube.com/watch?v=ml6cT4AZdqI"),
        "Downward Dog": ("🐶", "https://www.youtube.com/watch?v=0Fx8R9l2cOQ"),
        "Cobra Pose": ("🐍", "https://www.youtube.com/watch?v=JDcdhTuycOI"),
        "Dynamic Stretching": ("🤸", "https://www.youtube.com/watch?v=3KquFZYi6L0"),
        "Foam Rolling": ("🧽", "https://www.youtube.com/watch?v=8caF1Keg2XU"),
        "Kettlebell Swings": ("🏋️‍♂️", "https://www.youtube.com/watch?v=6kALZikXxLc"),
        "Battle Ropes": ("🪢", "https://www.youtube.com/watch?v=Qv6jG6bB9i4"),
        "Burpees": ("🏃‍♂️", "https://www.youtube.com/watch?v=TU8QYVW0gDU"),
        "Medicine Ball Slams": ("🏐", "https://www.youtube.com/watch?v=F1C2R1j4Zxk")
    }

def generate_workout_plan(goal, experience):
    # Use a subset of the exercise library for each goal
    library = get_exercise_library()
    plans = {
        "Lose Weight": ["Burpees", "Jump Rope", "HIIT Circuits", "Treadmill Running", "Cycling"],
        "Build Muscle": ["Bench Press", "Push-Ups", "Pull-Ups", "Squats", "Deadlifts", "Bicep Curl", "Tricep Dips"],
        "Increase Endurance": ["Rowing Machine", "Jump Rope", "Treadmill Running", "Planks", "Lunges"],
        "General Fitness": ["Push-Ups", "Squats", "Planks", "Dynamic Stretching", "Foam Rolling"]
    }
    reps = {"Beginner": "10 reps", "Intermediate": "15 reps", "Advanced": "20 reps"}
    plan_names = plans.get(goal, plans["General Fitness"])
    return [(f"{library[name][0]} {name}", library[name][1], reps[experience]) for name in plan_names]

def main():
    st.title("Personal Fitness Tracker")
    st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as `Age`, `Gender`, `BMI`, etc., into this WebApp and then you will see the predicted value of kilocalories burned.")
    # Get user input and goal
    df_user, goal, experience = user_input_features()

    # --- Interactive Level Selector with Horizontal Scrollable Plan ---
    st.write("---")
    st.markdown("""
    <h2 style='text-align:center;color:#ff4b4b;font-family:sans-serif;'>🔥 Choose Your Challenge! 🔥</h2>
    <p style='text-align:center;color:#444;font-size:1.1em;'>Select your level and get inspired to push your limits!</p>
    """, unsafe_allow_html=True)

    level_options = ["Beginner", "Pro", "Elite"]
    level_exercises = {
        "Beginner": ["Push-Ups", "Squats", "Planks", "Dynamic Stretching", "Foam Rolling"],
        "Pro": ["Bench Press", "Pull-Ups", "Deadlifts", "Bicep Curl", "Tricep Dips", "Lunges", "Lat Pulldown"],
        "Elite": ["Arnold Press", "Battle Ropes", "HIIT Circuits", "Medicine Ball Slams", "Kettlebell Swings", "Rowing Machine", "Treadmill Running"]
    }
    reps = {"Beginner": "10 reps", "Pro": "15 reps", "Elite": "20 reps"}
    library = get_exercise_library()

    selected_level = st.selectbox("Select your workout level:", level_options, index=0, key="level_select")
    st.markdown(f"<h3 style='color:#1f77b4;font-family:sans-serif;text-align:center;'>{selected_level} Level</h3>", unsafe_allow_html=True)

    # --- Custom CSS for horizontal scroll and 3D hover effect ---
    st.markdown("""
    <style>
    .scrolling-wrapper {
      overflow-x: auto;
      white-space: nowrap;
      padding-bottom: 24px; /* For 3D hover */
      padding-top: 18px;   /* Added for top edge */
      margin-bottom: 8px;
      padding-left: 32px;  /* Increased for left edge */
      padding-right: 32px; /* Increased for right edge */
      overflow-y: visible !important;
    }
    .workout-card {
      display: inline-block;
      width: 200px;
      margin-right: 12px;
      vertical-align: top;
      background: linear-gradient(135deg,#f0f2f6 60%,#e0e7ef 100%);
      border-radius: 18px;
      padding: 18px 8px 14px 8px;
      box-shadow: 0 2px 8px #0001;
      text-align: center;
      transition: transform 0.25s cubic-bezier(.25,.8,.25,1), box-shadow 0.25s;
      cursor: pointer;
      position: relative;
      z-index: 1;
      overflow: visible;
    }
    .workout-card:hover {
      transform: scale3d(1.08,1.08,1.08) rotateY(8deg) rotateX(2deg);
      box-shadow: 0 8px 32px #1f77b455, 0 1.5px 8px #ff4b4b33;
      z-index: 10;
    }
    .workout-card.selected {
      border: 4.5px solid #ff4b4b;
      box-shadow: 0 8px 32px #ff4b4b55, 0 1.5px 8px #1f77b433;
      z-index: 11;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Card selection logic ---
    if 'selected_card' not in st.session_state:
        st.session_state['selected_card'] = None

    # Render the horizontal scrollable workout plan with 3D hover and selection
    cards_html = ["<div class='scrolling-wrapper'>"]
    for ex in level_exercises[selected_level]:
        emoji, url = library[ex]
        selected = (st.session_state['selected_card'] == ex)
        card_class = "workout-card selected" if selected else "workout-card"
        cards_html.append(
            f"<div class='{card_class}' onclick=\"window.parent.postMessage({{type: 'select_card', card: '{ex}'}}, '*')\">"
            f"<a href='{url}' target='_blank' style='text-decoration:none;color:inherit;'>"
            f"<span style='font-size:3.2em;display:inline-block;width:70px;height:70px;line-height:70px;border-radius:50%;border:5px solid #ff4b4b;background:#fff;box-sizing:border-box;margin-bottom:8px;overflow:hidden;'>{emoji}</span><br>"
            f"<b style='font-size:1.1em;color:#222'>{ex}</b><br>"
            f"<span style='color:#ff4b4b;font-weight:bold;'>{reps[selected_level]}</span><br>"
            f"<span style='color:#1f77b4;text-decoration:underline;font-size:1.1em;'>🎬 Video</span>"
            f"</a>"
            f"</div>"
        )
    cards_html.append("</div>")
    st.markdown("\n".join(cards_html), unsafe_allow_html=True)

    # --- JS to handle card selection and communicate with Streamlit ---
    st.markdown("""
    <script>
    window.addEventListener('message', (event) => {
      if (event.data && event.data.type === 'select_card') {
        const card = event.data.card;
        const streamlitDoc = window.parent.document;
        const input = streamlitDoc.querySelector('input[data-testid="stTextInput"]');
        if (input) {
          input.value = card;
          input.dispatchEvent(new Event('input', { bubbles: true }));
        }
        window.parent.postMessage({ type: 'streamlit:setComponentValue', value: card }, '*');
      }
    });
    </script>
    """, unsafe_allow_html=True)

    # --- Show details of selected card ---
    st.markdown(f"<p style='color:#888;font-size:0.95em;text-align:center;'>Keep going! Every rep makes you stronger. 💯</p>", unsafe_allow_html=True)
    st.write("---")

    st.write("## Workout Session Log")
    if "workout_log" not in st.session_state:
        st.session_state["workout_log"] = []
    if "workout_start" not in st.session_state:
        st.session_state["workout_start"] = None
    if "workout_end" not in st.session_state:
        st.session_state["workout_end"] = None

    col_log1, col_log2 = st.columns(2)
    with col_log1:
        if st.session_state["workout_start"] is None:
            if st.button("Start Workout 🟢"):
                st.session_state["workout_start"] = time.strftime('%Y-%m-%d %H:%M:%S')
                st.success(f"Workout started at {st.session_state['workout_start']}")
        else:
            st.info(f"Workout started at {st.session_state['workout_start']}")
    with col_log2:
        if st.session_state["workout_start"] is not None and st.session_state["workout_end"] is None:
            if st.button("End Workout 🔴"):
                st.session_state["workout_end"] = time.strftime('%Y-%m-%d %H:%M:%S')
                # Calculate duration
                start_time = time.strptime(st.session_state["workout_start"], '%Y-%m-%d %H:%M:%S')
                end_time = time.strptime(st.session_state["workout_end"], '%Y-%m-%d %H:%M:%S')
                duration_sec = time.mktime(end_time) - time.mktime(start_time)
                duration_min = int(duration_sec // 60)
                st.session_state["workout_log"].append({
                    "goal": goal,
                    "experience": experience,
                    "start": st.session_state["workout_start"],
                    "end": st.session_state["workout_end"],
                    "duration": duration_min
                })
                st.success(f"Workout ended at {st.session_state['workout_end']} (Duration: {duration_min} min)")
                st.session_state["workout_start"] = None
                st.session_state["workout_end"] = None
        elif st.session_state["workout_end"] is not None:
            st.info(f"Workout ended at {st.session_state['workout_end']}")

    if st.session_state["workout_log"]:
        st.write("### Workout History 🗓️")
        for i, log in enumerate(st.session_state["workout_log"][::-1]):
            st.markdown(f"<b>Start:</b> {log['start']}<br><b>End:</b> {log['end']}<br><b>Duration:</b> {log['duration']} min<br><b>Goal:</b> <span style='color:#1f77b4'>{log['goal']}</span> (<i>{log['experience']}</i>)", unsafe_allow_html=True)
    st.write("---")
    # Load data
    try:
        exercise_df = load_data()
        X_train, y_train, X_test, y_test = prepare_model_data(exercise_df)
        model = train_model(X_train, y_train)
        
        # Display user parameters
        st.write("---")
        st.header("Your Parameters:")
        show_progress()
        st.write(df_user)
        
        # Align prediction data columns with training data
        df_user = df_user.reindex(columns=X_train.columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(df_user)
        
        # Display prediction
        st.write("---")
        st.header("Prediction:")
        show_progress()
        st.metric("Calories Burned", f"{round(prediction[0], 2)} kilocalories")
        
        # Display similar results
        st.write("---")
        st.header("Similar Results:")
        show_progress()
        
        # Find similar results based on predicted calories
        calorie_range = [prediction[0] - 10, prediction[0] + 10]
        similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & 
                                  (exercise_df["Calories"] <= calorie_range[1])]
        
        if not similar_data.empty:
            st.write(similar_data.sample(min(5, len(similar_data))))
        else:
            st.write("No similar results found in the dataset.")
        
        # Display general information
        st.write("---")
        st.header("General Information:")
        
        # Boolean logic for age, duration, etc., compared to the user's input
        boolean_age = (exercise_df["Age"] < df_user["Age"].values[0]).tolist()
        boolean_duration = (exercise_df["Duration"] < df_user["Duration"].values[0]).tolist()
        boolean_body_temp = (exercise_df["Body_Temp"] < df_user["Body_Temp"].values[0]).tolist()
        boolean_heart_rate = (exercise_df["Heart_Rate"] < df_user["Heart_Rate"].values[0]).tolist()
        
        # Show calories burned count (from prediction)
        st.metric("Calories Burned (Predicted)", f"{round(prediction[0], 2)} kcal", help="Predicted calories burned for your input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Age Percentile", f"{round(sum(boolean_age) / len(boolean_age) * 100, 1)}%", 
                     help="Percentage of people in the dataset who are younger than you")
            st.metric("Heart Rate Percentile", f"{round(sum(boolean_heart_rate) / len(boolean_heart_rate) * 100, 1)}%",
                     help="Percentage of people in the dataset with lower heart rate during exercise")
        
        with col2:
            st.metric("Duration Percentile", f"{round(sum(boolean_duration) / len(boolean_duration) * 100, 1)}%",
                     help="Percentage of people in the dataset with shorter exercise duration")
            st.metric("Body Temperature Percentile", f"{round(sum(boolean_body_temp) / len(boolean_body_temp) * 100, 1)}%",
                     help="Percentage of people in the dataset with lower body temperature during exercise")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please make sure the data files 'calories.csv' and 'exercise.csv' are in the same directory as this app.")

if __name__ == "__main__":
    main()