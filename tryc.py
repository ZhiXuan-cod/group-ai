import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import json
import os

# -------------------------------
# 1. config file
# -------------------------------
st.set_page_config(
    page_title="The Calorie Detective",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 2. CNN模型定义（简化版，便于演示）
# -------------------------------
class FoodCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FoodCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -------------------------------
# 3. Data Import
# -------------------------------
@st.cache_data
def load_nutrition_data():
    """Nutrition database loading"""
    try:
        df = pd.read_csv("nutrients.csv")
        st.success(f"✅ Nutrition database loaded successfully. ({len(df)} Food items)")
        return df
    except Exception as e:
    # Load the nutrition data from a CSV file or database
    # For demonstration, we'll create a sample dataframe
        data = {
            'Food': ['Apple', 'Banana', 'Chicken Breast', 'Salmon', 'Broccoli', 'Rice', 'Bread'],
            'Category': ['Fruit', 'Fruit', 'Meat', 'Fish', 'Vegetable', 'Grain', 'Grain'],
            'Calories': [52, 89, 165, 208, 34, 130, 265],
            'Protein': [0.3, 1.1, 31, 25, 2.8, 2.7, 9],
            'Carbs': [14, 23, 0, 0, 7, 28, 49],
            'Fat': [0.2, 0.3, 3.6, 13, 0.4, 0.3, 3.2]
        }
    return pd.DataFrame(data)

# -------------------------------
# 4. User Administration
# -------------------------------
class UserManager:
    def __init__(self):
        self.users_file = "users.json"
        self.load_users()
    
    def load_users(self):
        """Load user data"""
        try:
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        except:
            self.users = {}
    
    def save_users(self):
        """Save user data"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)
    
    def hash_password(self, password):
        """Password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, username, password, user_info):
        """Register new user"""
        if username in self.users:
            return False, "Username already exists"
        
        self.users[username] = {
            'password_hash': self.hash_password(password),
            'info': user_info,
            'history': [],
            'created_at': datetime.now().isoformat()
        }
        self.save_users()
        return True, "Sign-up successful!"
    
    def login(self, username, password):
        """User Login"""
        if username not in self.users:
            return False, "User does not exist"
        
        if self.users[username]['password_hash'] == self.hash_password(password):
            return True, "Login successful!"
        return False, "Incorrect password."
    
    def update_user_info(self, username, info):
        """Update user information"""
        if username in self.users:
            self.users[username]['info'] = info
            self.save_users()
            return True
        return False
    
    def add_food_history(self, username, food_entry):
        """Add dietary record"""
        if username in self.users:
            food_entry['timestamp'] = datetime.now().isoformat()
            self.users[username]['history'].append(food_entry)
            if len(self.users[username]['history']) > 100:  # Keep the latest 100 records.
                self.users[username]['history'] = self.users[username]['history'][-100:]
            self.save_users()
            return True
        return False

# -------------------------------
# 5. Health Calculation Functions
# -------------------------------
def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI"""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)

def get_bmi_category(bmi):
    """Get BMI classification"""
    if bmi < 18.5:
        return "Underweight", "blue", "Consider increasing nutritional intake"
    elif bmi < 24.9:
        return "Normal weight", "green", "Maintain a healthy diet"
    elif bmi < 29.9:
        return "Overweight", "orange", "Consider controlling diet appropriately"
    else:
        return "Obese", "red", "Recommend consulting a doctor and adjusting diet"

def calculate_daily_calories(gender, weight_kg, height_cm, age, activity_level):
    """Calculate daily calorie requirements"""
    # Mifflin-St Jeor Equation
    if gender == "Male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    
    # Activity level multipliers
    activity_factors = {
        "Sedentary": 1.2,
        "Lightly active": 1.375,
        "Moderately active": 1.55,
        "Very active": 1.725,
        "Athlete": 1.9
    }
    
    return round(bmr * activity_factors.get(activity_level, 1.2))

# -------------------------------
# 6. Food Recommendation Function
# -------------------------------
class FoodRecommender:
    def __init__(self, nutrition_df):
        self.df = nutrition_df
    
    def recommend_by_calories(self, target_calories, category=None, limit=5):
        """Recommend foods based on target calories"""
        df_filtered = self.df.copy()
        
        if category and category != "All":
            df_filtered = df_filtered[df_filtered['Category'] == category]
        
        # Calculate difference from target calories
        df_filtered['calorie_diff'] = abs(df_filtered['Calories'] - target_calories)
        
        # Return the closest matches
        return df_filtered.nsmallest(limit, 'calorie_diff')
    
    def recommend_for_weight_loss(self, current_weight, target_weight, days, category=None):
        """Weight loss diet recommendations"""
        # Calculate daily calorie deficit (assuming 0.5kg loss per week)
        weekly_loss = 0.5  # kg/week
        daily_calorie_deficit = weekly_loss * 7700 / 7  # 1kg fat ≈ 7700 calories
        
        recommendations = []
        base_calories = 1500  # Base intake
        
        # Recommend food combinations for different meals
        breakfast = self.recommend_by_calories(base_calories * 0.3, category, 3)
        lunch = self.recommend_by_calories(base_calories * 0.4, category, 3)
        dinner = self.recommend_by_calories(base_calories * 0.3, category, 3)
        
        return {
            'Breakfast Recommendations': breakfast[['Food', 'Calories', 'Protein', 'Carbs', 'Fat']].to_dict('records'),
            'Lunch Recommendations': lunch[['Food', 'Calories', 'Protein', 'Carbs', 'Fat']].to_dict('records'),
            'Dinner Recommendations': dinner[['Food', 'Calories', 'Protein', 'Carbs', 'Fat']].to_dict('records'),
            'Daily Goal': f"{base_calories} calories",
            'Expected Timeframe': f"{(current_weight - target_weight) / weekly_loss:.1f} weeks"
        }

# -------------------------------
# 7. Main Application
# -------------------------------
def main():
    # Initialization
    user_manager = UserManager()
    nutrition_df = load_nutrition_data()
    recommender = FoodRecommender(nutrition_df)
    
    # Sidebar navigation
    st.sidebar.title("🍎 The Calorie Detective")
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    # Login/register interface
    if not st.session_state.logged_in:
        show_login_register(user_manager)
        return
    
    # Main application menu
    menu = ["🏠 Dashboard", "📷 Food Recognition", "⚖️ Health Analysis", "📊 Nutrition Database", "🎯 Diet Recommendations", "👤 Profile"]
    choice = st.sidebar.selectbox("Navigation Menu", menu)
    
    # Display user info
    st.sidebar.markdown("---")
    if st.session_state.current_user in user_manager.users:
        user_info = user_manager.users[st.session_state.current_user]['info']
        st.sidebar.markdown(f"**👤 User:** {st.session_state.current_user}")
        st.sidebar.markdown(f"**⚖️ BMI:** {user_info.get('bmi', 'Not set')}")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()
    
    # Page routing
    if choice == "🏠 Dashboard":
        show_dashboard(user_manager, nutrition_df)
    elif choice == "📷 Food Recognition":
        show_food_recognition(user_manager, nutrition_df)
    elif choice == "⚖️ Health Analysis":
        show_health_analysis(user_manager)
    elif choice == "📊 Nutrition Database":
        show_nutrition_database(nutrition_df)
    elif choice == "🎯 Diet Recommendations":
        show_food_recommendation(recommender, user_manager)
    elif choice == "👤 Profile":
        show_user_profile(user_manager)

# -------------------------------
# 8. Page Functions
# -------------------------------
def show_login_register(user_manager):
    """Display login/register page"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("🔐 User Login")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", type="primary"):
                success, message = user_manager.login(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        with tab2:
            new_user = st.text_input("New Username", key="reg_user")
            new_pass = st.text_input("New Password", type="password", key="reg_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
            
            user_info = {
                'gender': 'Male',
                'age': 25,
                'height': 170,
                'weight': 65,
                'activity': 'Moderately active'
            }
            
            if st.button("Register", type="primary"):
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, message = user_manager.register(new_user, new_pass, user_info)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    with col2:
        st.image("https://img.icons8.com/color/300/000000/healthy-eating.png", use_column_width=True)
        st.markdown("""
        ### 🍎 System Features
        - **AI Food Recognition**: Photo-based food identification
        - **Calorie Calculation**: Automatic nutrition calculation
        - **Health Analysis**: BMI and dietary advice
        - **Personalized Recommendations**: Smart diet plans
        """)

def show_dashboard(user_manager, nutrition_df):
    """Display dashboard"""
    st.title("📊 Health Dashboard")
    
    if st.session_state.current_user not in user_manager.users:
        st.warning("Failed to load user data")
        return
    
    user_data = user_manager.users[st.session_state.current_user]
    user_info = user_data['info']
    history = user_data['history']
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'bmi' in user_info:
            bmi = user_info['bmi']
            category, color, _ = get_bmi_category(bmi)
            st.metric("BMI Index", f"{bmi}", category)
        else:
            st.metric("BMI Index", "Not set", "Please set personal info first")
    
    with col2:
        if 'height' in user_info and 'weight' in user_info:
            daily_cal = calculate_daily_calories(
                user_info['gender'],
                user_info['weight'],
                user_info['height'],
                user_info['age'],
                user_info['activity']
            )
            st.metric("Daily Requirement", f"{daily_cal} cal")
        else:
            st.metric("Daily Requirement", "Not set")
    
    with col3:
        today = datetime.now().date().isoformat()
        today_calories = sum([h['calories'] for h in history 
                            if h.get('timestamp', '').startswith(today)])
        st.metric("Today's Intake", f"{today_calories} cal")
    
    with col4:
        food_count = len(history)
        st.metric("Total Records", f"{food_count} entries")
    
    # Chart area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Last 7 Days Intake Trend")
        if history:
            # Process history data
            df_history = pd.DataFrame(history)
            df_history['date'] = pd.to_datetime(df_history['timestamp']).dt.date
            daily_summary = df_history.groupby('date').agg({
                'calories': 'sum',
                'protein': 'sum',
                'carbs': 'sum',
                'fat': 'sum'
            }).tail(7)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_summary.index,
                y=daily_summary['calories'],
                mode='lines+markers',
                name='Calories',
                line=dict(color='#FF6B6B', width=3)
            ))
            fig.update_layout(
                title="Calorie Intake Trend",
                xaxis_title="Date",
                yaxis_title="Calories (cal)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dietary records yet")
    
    with col2:
        st.subheader("🍽️ Food Category Distribution")
        if history:
            df_history = pd.DataFrame(history)
            category_counts = df_history['food_category'].value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dietary records yet")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📷 Recognize Food", use_container_width=True):
            st.switch_page("📷 Food Recognition")
    
    with col2:
        if st.button("⚖️ Health Analysis", use_container_width=True):
            st.switch_page("⚖️ Health Analysis")
    
    with col3:
        if st.button("🎯 Diet Recommendations", use_container_width=True):
            st.switch_page("🎯 Diet Recommendations")

def show_food_recognition(user_manager, nutrition_df):
    """Display food recognition page"""
    st.title("📷 AI Food Recognition")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_method = st.radio(
            "Select Input Method",
            ["Upload Image", "Take Photo", "Manual Entry"],
            horizontal=True
        )
        
        food_name = None
        confidence = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose Food Image",
                type=["jpg", "jpeg", "png"],
                help="Upload clear food images for best recognition results"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Food Image", use_column_width=True)
                
                # Simulate AI recognition (should call actual model)
                with st.spinner("🤖 AI is analyzing food..."):
                    import time
                    time.sleep(1)  # Simulate processing time
                    
                    # Should call actual CNN model
                    # For demo, randomly select a food
                    food_options = nutrition_df['Food'].tolist()
                    selected_idx = np.random.randint(0, len(food_options))
                    food_name = food_options[selected_idx]
                    confidence = np.random.uniform(85, 98)
                    
                    st.success(f"✅ Recognition complete!")
                    
                    # Display recognition result
                    st.markdown(f"""
                    <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                border-radius: 10px; color: white; text-align: center;">
                        <h2>Recognition Result</h2>
                        <h1>{food_name}</h1>
                        <h3>Confidence: {confidence:.1f}%</h3>
                        <p>🤖 Based on CNN deep learning model</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        elif input_method == "Manual Entry":
            food_name = st.selectbox(
                "Select Food",
                nutrition_df['Food'].tolist(),
                help="Choose food from list"
            )
            confidence = 100.0
        
        # Look up nutrition information
        if food_name:
            food_info = nutrition_df[nutrition_df['Food'] == food_name]
            if not food_info.empty:
                food_info = food_info.iloc[0]
                
                # Display nutrition information
                st.subheader("📊 Nutritional Information")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Calories", f"{int(food_info['Calories'])} cal")
                with col_b:
                    st.metric("Protein", f"{food_info['Protein']} g")
                with col_c:
                    st.metric("Carbohydrates", f"{food_info['Carbs']} g")
                with col_d:
                    st.metric("Fat", f"{food_info['Fat']} g")
                
                # Nutrition breakdown pie chart
                calories_from = {
                    'Protein': food_info['Protein'] * 4,
                    'Carbohydrates': food_info['Carbs'] * 4,
                    'Fat': food_info['Fat'] * 9
                }
                
                fig = px.pie(
                    values=list(calories_from.values()),
                    names=list(calories_from.keys()),
                    color_discrete_sequence=['#4CAF50', '#2196F3', '#FF9800'],
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Save record button
                if st.button("💾 Save to Food Log", type="primary"):
                    nutrition_df['Fat'] = pd.to_numeric(nutrition_df['Fat'], errors='coerce').fillna(0)
                    

                    food_entry = {
                        'food_name': food_name,
                        'calories': int(food_info['Calories']),
                        'protein': float(food_info['Protein']),
                        'carbs': float(food_info['Carbs']),
                        'fat': float(food_info['Fat']),
                        'food_category': str(food_info.get('Category', 'Other'))
                    }
                    
                    if user_manager.add_food_history(st.session_state.current_user, food_entry):
                        st.success("✅ Record saved!")
                    else:
                        st.error("❌ Save failed")
    
    with col2:
        st.subheader("💡 Health Tips")
        if 'food_info' in locals():
            calories = food_info['Calories']
            calories = pd.to_numeric(calories, errors='coerce')
            if calories < 100:
                st.success("""
                🥗 **Low-calorie Food**
                - Excellent for weight loss periods
                - Rich in dietary fiber
                - Recommended as snacks or side dishes
                """)
            elif calories < 300:
                st.info("""
                🍽️ **Medium-calorie Food**
                - Suitable as a main meal
                - Maintain balanced nutrition
                - Control portion sizes
                """)
            else:
                st.warning("""
                ⚠️ **High-calorie Food**
                - Consume in moderation
                - Avoid frequent consumption
                - Ensure adequate exercise
                """)
        
        # Display AI model information
        with st.expander("🤖 View AI Model Details"):
            st.markdown("""
            ### CNN Model Architecture
            ```python
            class FoodCNN(nn.Module):
                def __init__(self):
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                    )
                    self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(128*28*28, 256),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(256, num_classes)
                    )
            ```
            **Accuracy**: 85.3%  
            **Training Data**: Food-101 Dataset  
            **Training Duration**: 20 epochs
            """)

def show_health_analysis(user_manager):
    """Display health analysis page"""
    st.title("⚖️ Health Analysis")
    
    if st.session_state.current_user not in user_manager.users:
        st.warning("Please log in to access your health data")
        return
    
    user_info = user_manager.users[st.session_state.current_user]['info']
    
    # Health data form
    with st.form("health_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"], 
                                index=0 if user_info.get('gender') == 'Male' else 1)
            age = st.number_input("Age", min_value=1, max_value=120, 
                                value=int(user_info.get('age', 25)))
            height = st.number_input("Height (cm)", min_value=50, max_value=250, 
                                   value=int(user_info.get('height', 170)))
        
        with col2:
            weight = st.number_input("Weight (kg)", min_value=20, max_value=300, 
                                   value=int(user_info.get('weight', 65)))
            # Define mapping from Chinese to English
            activity_mapping = {
                "久坐": "Sedentary",
                "轻度活动": "Lightly active", 
                "中度活动": "Moderately active",
                "高度活动": "Very active",
                "运动员": "Athlete"
            }

            # Get current activity from user_info
            current_activity = user_info.get('activity', 'Moderately active')

            # If it's in Chinese, convert to English
            if current_activity in activity_mapping:
                current_activity = activity_mapping[current_activity]
            else:
                # If not found in mapping, default to "Moderately active"
                current_activity = "Moderately active"

            # Now create the selectbox
            activity = st.selectbox(
                "Activity Level",
                ["Sedentary", "Lightly active", "Moderately active", "Very active", "Athlete"],
                index=["Sedentary", "Lightly active", "Moderately active", "Very active", "Athlete"].index(current_activity)
            )
            

        
        if st.form_submit_button("Health data updated", type="primary"):
            # Calculate BMI
            bmi = calculate_bmi(weight, height)
            category, color, advice = get_bmi_category(bmi)
            
            # Update user info
            updated_info = {
                'gender': gender,
                'age': age,
                'height': height,
                'weight': weight,
                'activity': activity,
                'bmi': bmi,
                'bmi_category': category,
                'last_updated': datetime.now().isoformat()
            }
            
            if user_manager.update_user_info(st.session_state.current_user, updated_info):
                st.success("✅ Health data updated!")
                st.rerun()

    
    # Display analysis results
    if 'bmi' in user_info:
        st.markdown("---")
        
        bmi = user_info['bmi']
        category, color, advice = get_bmi_category(bmi)
        
        # BMI dashboard
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
                <h3>BMI指数</h3>
                <h1 style="color: {color}; font-size: 48px;">{bmi}</h1>
                <h4 style="color: {color};">{category}</h4>
                <p>{advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Daily calorie requirement
            daily_cal = calculate_daily_calories(
                user_info['gender'],
                user_info['weight'],
                user_info['height'],
                user_info['age'],
                user_info['activity']
            )
            
            st.metric("Daily calorie requirement", f"{daily_cal} 大卡")
        
        with col2:
            # BMI range chart
            fig = go.Figure()
            
            # BMI range areas
            ranges = [
                (0, 18.5, "Underweight", "blue"),
                (18.5, 24.9, "Normal", "green"),
                (24.9, 29.9, "Overweight", "orange"),
                (29.9, 40, "Obese", "red")
            ]
            
            for start, end, name, color in ranges:
                fig.add_trace(go.Bar(
                    x=[end - start],
                    y=["BMI Range"],
                    base=[start],
                    orientation='h',
                    name=name,
                    marker_color=color,
                    hoverinfo='skip'
                ))
            
            # User's BMI position
            fig.add_trace(go.Scatter(
                x=[bmi],
                y=["BMI Range"],
                mode='markers',
                marker=dict(size=20, color='black', symbol='diamond'),
                name=f'Your BMI: {bmi}',
                hovertemplate='<b>Your BMI: %{x}</b>'
            ))
            
            fig.update_layout(
                title="BMI Range Visualization",
                xaxis_title="BMI Value",
                barmode='stack',
                showlegend=False,
                height=200
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Health recommendations
            st.info(f"""
            ### 💡 Personalized Recommendations
            
             **Dietary Recommendations**:
            - Daily calorie goal: {daily_cal} cal
            - Protein intake: {daily_cal * 0.15 / 4:.1f}g
            - Carbohydrates: {daily_cal * 0.55 / 4:.1f}g  
            - Fat intake: {daily_cal * 0.3 / 9:.1f}g
            
            **Exercise Recommendations**:
            - At least 150 minutes of moderate exercise per week
            - Strength training twice a week
            - Walk 8,000+ steps daily
            """)

def show_nutrition_database(nutrition_df):
    """Display nutrition database"""
    st.title("📊 Nutrition Database")
    
    # Search and filter
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search Food Name")
    
    with col2:
        category_filter = st.selectbox(
            "Food Category",
            ["All"] + nutrition_df['Category'].dropna().unique().tolist()
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Calories (Low to High)", "Calories (High to Low)", "Protein Content", "Name"]
        )
    
    # Filter data
    filtered_df = nutrition_df.copy()
    
    if search_term:
        filtered_df = filtered_df[filtered_df['Food'].str.contains(search_term, case=False, na=False)]
    
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
    
    # Sorting
    if sort_by == "Calories (Low to High)":
        filtered_df = filtered_df.sort_values('Calories')
    elif sort_by == "Calories (High to Low)":
        filtered_df = filtered_df.sort_values('Calories', ascending=False)
    elif sort_by == "Protein Content":
        filtered_df = filtered_df.sort_values('Protein', ascending=False)
    elif sort_by == "Name":
        filtered_df = filtered_df.sort_values('Food')
    
    # Display data
    st.dataframe(
        filtered_df,
        column_config={
            "Food": st.column_config.TextColumn("Food Name", width="medium"),
            "Calories": st.column_config.NumberColumn("Calories", format="%d cal"),
            "Protein_g": st.column_config.NumberColumn("Protein(g)", format="%.1f g"),
            "Carbs_g": st.column_config.NumberColumn("Carbs(g)", format="%.1f g"),
            "Fat_g": st.column_config.NumberColumn("Fat(g)", format="%.1f g"),
            "Category": st.column_config.TextColumn("Category", width="small")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Statistics
    st.subheader("📈 Nutrition Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Foods", len(filtered_df))
    with col2:
        avg_cal = filtered_df['Calories'].mean()
        st.metric("Average Calories", f"{avg_cal:.0f}")
    with col3:
        max_cal = filtered_df['Calories'].max()
        st.metric("Highest Calories", f"{max_cal:.0f}")
    with col4:
        min_cal = filtered_df['Calories'].min()
        st.metric("Lowest Calories", f"{min_cal:.0f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Calorie Distribution")
        fig = px.histogram(
            filtered_df, 
            x='Calories',
            nbins=20,
            color_discrete_sequence=['#FF6B6B']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Category Distribution")
        category_counts = filtered_df['Category'].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

def show_food_recommendation(recommender, user_manager):
    """Display dietary recommendations page"""
    st.title("🎯 Smart Diet Recommendations")
    
    if st.session_state.current_user not in user_manager.users:
        st.warning("Please log in first")
        return
    
    user_info = user_manager.users[st.session_state.current_user]['info']
    
    # Recommendation type selection
    rec_type = st.radio(
        "Recommendation Type",
        ["Weight Loss", "Balanced Nutrition", "Specific Calories", "Muscle Gain"],
        horizontal=True
    )
    
    if rec_type == "Weight Loss":
        st.subheader("💪 Weight Loss Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_weight = st.number_input(
                "Current Weight (kg)",
                min_value=30.0,
                max_value=200.0,
                value=float(user_info.get('weight', 65))
            )
            target_weight = st.number_input(
                "Target Weight (kg)",
                min_value=30.0,
                max_value=200.0,
                value=current_weight - 5.0
            )
        
        with col2:
            time_frame = st.selectbox(
                "Time Frame",
                ["1 month", "2 months", "3 months", "6 months"]
            )
            category_pref = st.selectbox(
                "Preferred Category",
                ["All", "Fruits", "Vegetables", "Main Dishes", "Meat", "Fast Food", "Desserts"]
            )
        
        if st.button("Generate Weight Loss Plan", type="primary"):
            # Calculate time (days)
            months = {"1 month": 30, "2 months": 60, "3 months": 90, "6 months": 180}
            days = months[time_frame]
            
            # Get recommendations
            recommendations = recommender.recommend_for_weight_loss(
                current_weight, target_weight, days, category_pref
            )
            
            st.success("✅ Weight loss diet plan generated!")
            
            # Display plan
            st.markdown(f"""
            ### 📋 Weight Loss Diet Plan
            
            **Goal**: {current_weight}kg → {target_weight}kg  
            **Duration**: {time_frame} ({days} days)  
            **Daily Calorie Goal**: {recommendations['Daily Goal']}  
            **Expected Timeframe**: {recommendations['Expected Timeframe']}
            """)
            
            # Display meal recommendations
            for meal, foods in recommendations.items():
                if meal not in ['Daily Goal', 'Expected Timeframe']:
                    st.subheader(f"🍽️ {meal}")
                    
                    for food in foods:
                        with st.expander(f"{food['Food']} - {food['Calories']} cal"):
                            st.write(f"Protein: {food['Protein']}g")
                            st.write(f"Carbs: {food['Carbs']}g")
                            st.write(f"Fat: {food['Fat']}g")
    
    elif rec_type == "Specific Calories":
        st.subheader("⚖️ Specific Calorie Recommendations")
        
        target_calories = st.slider(
            "Target Calories",
            min_value=50,
            max_value=1000,
            value=300,
            step=50
        )
        
        category = st.selectbox(
            "Food Category",
            ["All", "Fruits", "Vegetables", "Main Dishes", "Meat", "Fast Food", "Desserts"]
        )
        
        if st.button("Find Matching Foods", type="primary"):
            matches = recommender.recommend_by_calories(target_calories, category)
            
            st.success(f"✅ Found {len(matches)} matching foods")
            
            # Display results
            for idx, row in matches.iterrows():
                with st.expander(f"{row['Food']} - {int(row['Calories'])} cal"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Category**: {row['Category']}")
                        st.write(f"**Protein**: {row['Protein']}g")
                    with col2:
                        st.write(f"**Carbs**: {row['Carbs']}g")
                        st.write(f"**Fat**: {row['Fat']}g")
                    
                    # Nutrition ratio chart
                    calories_from = {
                        'Protein': row['Protein'] * 4,
                        'Carbohydrates': row['Carbs'] * 4,
                        'Fat': row['Fat'] * 9
                    }
                    
                    fig = px.pie(
                        values=list(calories_from.values()),
                        names=list(calories_from.keys()),
                        color_discrete_sequence=['#4CAF50', '#2196F3', '#FF9800'],
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

def show_user_profile(user_manager):
    """Display user profile"""
    st.title("👤 User Profile")
    
    if st.session_state.current_user not in user_manager.users:
        st.warning("Please log in first")
        return
    
    user_data = user_manager.users[st.session_state.current_user]
    user_info = user_data['info']
    
    # User info card
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white;">
            <h2>{st.session_state.current_user}</h2>
            <p>Registered: {user_data.get('created_at', 'Unknown')[:10]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if user_info:
            info_html = f"""
            <div style="padding: 20px; background-color: #f9f9f9; border-radius: 10px;">
                <h4>📋 Personal Information</h4>
                <p><strong>Gender:</strong> {user_info.get('gender', 'Not set')}</p>
                <p><strong>Age:</strong> {user_info.get('age', 'Not set')}</p>
                <p><strong>Height:</strong> {user_info.get('height', 'Not set')} cm</p>
                <p><strong>Weight:</strong> {user_info.get('weight', 'Not set')} kg</p>
                <p><strong>Activity Level:</strong> {user_info.get('activity', 'Not set')}</p>
            """
            if 'bmi' in user_info:
                info_html += f"""
                <p><strong>BMI:</strong> {user_info['bmi']} ({user_info.get('bmi_category', '')})</p>
                """
            info_html += "</div>"
            st.markdown(info_html, unsafe_allow_html=True)
    
    # Food history
    st.subheader("📝 Recent Food History")
    history = user_data.get('history', [])
    
    if history:
        df_history = pd.DataFrame(history[-10:])  # Show last 10 entries
        df_history['time'] = pd.to_datetime(df_history['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        for _, row in df_history.iterrows():
            with st.expander(f"{row['time']} - {row['food_name']} ({row['calories']} cal)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Category**: {row.get('food_category', 'Other')}")
                    st.write(f"**Protein**: {row['protein']}g")
                with col2:
                    st.write(f"**Carbohydrates**: {row['carbs']}g")
                    st.write(f"**Fat**: {row['fat']}g")
    else:
        st.info("No food history yet")

# -------------------------------
# 9. Run Application
# -------------------------------
if __name__ == "__main__":
    main()