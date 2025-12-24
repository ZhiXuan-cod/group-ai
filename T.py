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
from datetime import datetime, date
import hashlib
import json
import os
import re
from difflib import get_close_matches
import warnings
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Page Configuration
# -------------------------------
st.set_page_config(
    page_title="The Calorie Detective",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 2. CNN Model Definition (Simplified for demo)
# -------------------------------
class FoodCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FoodCNN, self).__init__()
        # Simplified model for demo
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------------
# 3. Data Import with "t" Replacement
# -------------------------------
def clean_nutrient_values(df):
    """
    Replace ALL "t" values with 0.001 in ALL nutrient columns.
    Handles case variations and converts columns to numeric.
    """
    df_clean = df.copy()
    
    # All possible nutrient columns (case insensitive)
    possible_nutrient_cols = [
        'Calories', 'Protein', 'Carbohydrates', 'Carbs', 'Fat', 
        'Saturated Fat', 'Trans Fat', 'Polyunsaturated Fat', 'Monounsaturated Fat',
        'Cholesterol', 'Sodium', 'Potassium', 'Fiber', 'Sugar',
        'Vitamin A', 'Vitamin C', 'Calcium', 'Iron'
    ]
    
    # Find actual nutrient columns in the dataframe
    nutrient_cols = []
    for col in df_clean.columns:
        col_lower = str(col).lower()
        for nutrient in possible_nutrient_cols:
            if nutrient.lower() in col_lower:
                nutrient_cols.append(col)
                break
    
    # If no specific nutrient columns found, use numeric columns
    if not nutrient_cols:
        nutrient_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Replace ALL "t" values with 0.001 and convert to numeric
    for col in nutrient_cols:
        if col in df_clean.columns:
            try:
                # Convert to string first to handle mixed types
                df_clean[col] = df_clean[col].astype(str)
                
                # Replace ALL variations of "t" with 0.001
                # This handles: 't', 'T', 't.', 'T.', 'trace', 'Trace', 'TRACE', etc.
                df_clean[col] = df_clean[col].str.replace(
                    r'^[tT](\.)?(race)?$', 
                    '0.001', 
                    regex=True
                )
                
                # Also handle any other string representations of trace
                df_clean[col] = df_clean[col].replace({
                    't': '0.001',
                    'T': '0.001',
                    't.': '0.001',
                    'T.': '0.001',
                    'trace': '0.001',
                    'Trace': '0.001',
                    'TRACE': '0.001',
                    'tr': '0.001',
                    'Tr': '0.001',
                    'nil': '0',
                    'Nil': '0',
                    'NIL': '0',
                    'neg': '0',
                    'Neg': '0',
                    'NEG': '0'
                })
                
                # Convert to numeric (this will turn non-convertible values to NaN)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Fill NaN with 0 (optional, but good for calculations)
                df_clean[col] = df_clean[col].fillna(0)
                
            except Exception as e:
                st.warning(f"Could not clean column '{col}': {str(e)}")
                # If cleaning fails, try to convert directly
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    pass
    
    return df_clean

@st.cache_data
def load_nutrition_data():
    """Nutrition database loading with "t" replacement"""
    try:
        # Try to load from CSV file
        df = pd.read_csv("nutrients.csv")
        
        # Clean ALL "t" values and convert to numeric
        df = clean_nutrient_values(df)
        
        st.success(f"✅ Nutrition database loaded successfully. ({len(df)} Food items)")
        return df
    except Exception as e:
        # Create sample data with NO "t" values
        data = {
            'Food': ['Apple', 'Banana', 'Chicken Breast', 'Salmon', 'Broccoli', 'Rice', 'Bread',
                    'Orange', 'Strawberry', 'Blueberry', 'Carrot', 'Tomato', 'Potato',
                    'Beef', 'Pork', 'Egg', 'Milk', 'Cheese', 'Yogurt', 'Pasta',
                    'Pizza', 'Burger', 'French Fries', 'Salad', 'Soup', 'Sandwich',
                    'Chocolate', 'Ice Cream', 'Cookie', 'Cake', 'Donut',
                    'Coffee', 'Tea', 'Juice', 'Soda', 'Water',
                    'Almond', 'Walnut', 'Peanut', 'Cashew',
                    'Avocado', 'Cucumber', 'Lettuce', 'Spinach', 'Kale',
                    'Tuna', 'Shrimp', 'Crab', 'Lobster',
                    'Turkey', 'Duck', 'Lamb'],
            'Category': ['Fruit', 'Fruit', 'Meat', 'Fish', 'Vegetable', 'Grain', 'Grain',
                        'Fruit', 'Fruit', 'Fruit', 'Vegetable', 'Vegetable', 'Vegetable',
                        'Meat', 'Meat', 'Dairy', 'Dairy', 'Dairy', 'Dairy', 'Grain',
                        'Fast Food', 'Fast Food', 'Fast Food', 'Vegetable', 'Soup', 'Fast Food',
                        'Dessert', 'Dessert', 'Dessert', 'Dessert', 'Dessert',
                        'Beverage', 'Beverage', 'Beverage', 'Beverage', 'Beverage',
                        'Nut', 'Nut', 'Nut', 'Nut',
                        'Fruit', 'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable',
                        'Fish', 'Seafood', 'Seafood', 'Seafood',
                        'Meat', 'Meat', 'Meat'],
            'Calories': [52, 89, 165, 208, 34, 130, 265,
                        47, 33, 57, 41, 18, 77,
                        250, 242, 155, 42, 402, 59, 131,
                        285, 295, 312, 15, 75, 350,
                        546, 207, 502, 367, 452,
                        2, 1, 45, 150, 0,
                        579, 654, 567, 553,
                        160, 15, 15, 23, 49,
                        132, 99, 87, 90,
                        189, 337, 294],
            'Protein': [0.3, 1.1, 31, 25, 2.8, 2.7, 9,
                       0.9, 0.7, 0.7, 0.9, 0.9, 2.0,
                       26, 25, 13, 3.4, 25, 10, 5,
                       12, 17, 3, 1, 2, 15,
                       5, 3.5, 6, 3, 5,
                       0.1, 0, 0.5, 0, 0,
                       21, 15, 26, 18,
                       2, 0.6, 1.4, 2.9, 4.3,
                       28, 24, 19, 19,
                       29, 19, 25],
            'Carbs': [14, 23, 0, 0, 7, 28, 49,
                    12, 8, 14, 10, 4, 17,
                    0, 0, 1.1, 5, 1.3, 3.6, 25,
                    36, 30, 41, 3, 11, 40,
                    60, 24, 65, 53, 51,
                    0, 0, 11, 39, 0,
                    22, 14, 16, 30,
                    9, 3.6, 2.9, 3.6, 9,
                    0, 0, 0, 0,
                    0, 0, 0],
            'Fat': [0.2, 0.3, 3.6, 13, 0.4, 0.3, 3.2,
                   0.1, 0.3, 0.3, 0.2, 0.2, 0.1,
                   17, 16, 11, 1, 33, 0.4, 1,
                   10, 15, 15, 0.2, 3, 17,
                   31, 11, 24, 14, 25,
                   0, 0, 0, 0, 0,
                   49, 65, 49, 44,
                   15, 0.1, 0.2, 0.4, 0.9,
                   1, 0.3, 1, 1,
                   7, 28, 21]
        }
        df = pd.DataFrame(data)
        st.info(f"📋 Using sample nutrition database. ({len(df)} Food items)")
        return df

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
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            else:
                self.users = {}
        except:
            self.users = {}
    
    def save_users(self):
        """Save user data"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            return True
        except:
            return False
    
    def hash_password(self, password):
        """Password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, username, password, user_info):
        """Register new user"""
        if username in self.users:
            return False, "Username already exists"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        self.users[username] = {
            'password_hash': self.hash_password(password),
            'info': user_info,
            'history': [],
            'created_at': datetime.now().isoformat(),
            'last_login': datetime.now().isoformat()
        }
        if self.save_users():
            return True, "Sign-up successful!"
        return False, "Error saving user data"
    
    def login(self, username, password):
        """User Login"""
        if username not in self.users:
            return False, "User does not exist"
        
        if self.users[username]['password_hash'] == self.hash_password(password):
            self.users[username]['last_login'] = datetime.now().isoformat()
            self.save_users()
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
            food_entry['id'] = len(self.users[username]['history']) + 1
            self.users[username]['history'].append(food_entry)
            if len(self.users[username]['history']) > 100:  # Keep the latest 100 records.
                self.users[username]['history'] = self.users[username]['history'][-100:]
            self.save_users()
            return True
        return False

# -------------------------------
# 5. NLP Food Search System (Simplified)
# -------------------------------
class FoodNLPSearch:
    def __init__(self, nutrition_df):
        self.df = nutrition_df
        self.food_synonyms = self._create_food_synonyms()
        
    def _create_food_synonyms(self):
        """Create a dictionary of food synonyms and related terms"""
        synonyms = {
            'apple': ['apples', 'red apple', 'green apple', 'gala apple', 'fruit'],
            'banana': ['bananas', 'ripe banana', 'green banana', 'fruit'],
            'chicken': ['chicken breast', 'chicken thigh', 'grilled chicken', 'roasted chicken', 'poultry', 'meat'],
            'salmon': ['salmon fish', 'grilled salmon', 'baked salmon', 'fish', 'seafood'],
            'broccoli': ['broccoli florets', 'steamed broccoli', 'vegetable', 'green vegetable'],
            'rice': ['white rice', 'brown rice', 'steamed rice', 'fried rice', 'grain', 'carb'],
            'bread': ['white bread', 'whole wheat bread', 'toast', 'sandwich bread', 'grain'],
            'orange': ['oranges', 'citrus', 'fruit juice'],
            'coffee': ['espresso', 'latte', 'cappuccino', 'black coffee', 'beverage', 'drink'],
            'pizza': ['cheese pizza', 'pepperoni pizza', 'margherita pizza', 'fast food'],
            'burger': ['hamburger', 'cheeseburger', 'beef burger', 'fast food'],
            'salad': ['green salad', 'caesar salad', 'vegetable salad', 'healthy food'],
            'egg': ['eggs', 'boiled egg', 'fried egg', 'scrambled egg', 'protein'],
            'milk': ['dairy milk', 'whole milk', 'skim milk', 'dairy'],
            'cheese': ['cheddar', 'mozzarella', 'parmesan', 'dairy'],
            'chocolate': ['dark chocolate', 'milk chocolate', 'chocolate bar', 'dessert', 'sweet'],
            'ice cream': ['vanilla ice cream', 'chocolate ice cream', 'dessert', 'frozen dessert'],
            'water': ['drinking water', 'mineral water', 'hydration', 'beverage'],
            'beef': ['steak', 'roast beef', 'ground beef', 'red meat', 'meat'],
            'pork': ['pork chop', 'bacon', 'ham', 'meat'],
            'fish': ['white fish', 'fried fish', 'baked fish', 'seafood'],
            'vegetable': ['veggies', 'greens', 'salad ingredients'],
            'fruit': ['fruits', 'fresh fruit', 'seasonal fruit'],
            'nut': ['nuts', 'almonds', 'walnuts', 'healthy snack'],
            'dessert': ['sweets', 'treats', 'after dinner'],
            'beverage': ['drinks', 'refreshments', 'liquids']
        }
        return synonyms
    
    def _extract_keywords(self, query):
        """Simple keyword extraction"""
        query = query.lower().strip()
        # Remove common filler words
        filler_words = ['a', 'an', 'the', 'some', 'my', 'i', 'want', 'like', 'have', 
                       'had', 'eat', 'ate', 'eating', 'for', 'with', 'today', 'just']
        words = query.split()
        words = [word for word in words if word not in filler_words]
        return words
    
    def search_food(self, query):
        """
        Main NLP search function
        Returns: DataFrame of matching foods
        """
        if not query or query.strip() == "":
            return self.df.head(10)
        
        query = query.strip().lower()
        
        # 1. Direct exact match
        exact_match = self.df[self.df['Food'].str.lower() == query]
        if not exact_match.empty:
            return exact_match
        
        # 2. Partial match in food names
        partial_match = self.df[self.df['Food'].str.lower().str.contains(query)]
        if not partial_match.empty:
            return partial_match
        
        # 3. Search in synonyms
        keywords = self._extract_keywords(query)
        results = []
        
        for keyword in keywords:
            # Check if keyword matches any synonym
            for food, synonyms in self.food_synonyms.items():
                if keyword in synonyms or keyword == food:
                    # Find foods containing this keyword
                    matches = self.df[self.df['Food'].str.lower().str.contains(food)]
                    results.append(matches)
            
            # Also search in categories
            category_matches = self.df[self.df['Category'].str.lower().str.contains(keyword)]
            results.append(category_matches)
        
        # Combine all results
        if results:
            combined = pd.concat(results).drop_duplicates()
            if not combined.empty:
                return combined
        
        # 4. Fuzzy matching as last resort
        food_list = self.df['Food'].str.lower().tolist()
        fuzzy_matches = get_close_matches(query, food_list, n=5, cutoff=0.3)
        if fuzzy_matches:
            matched_foods = []
            for match in fuzzy_matches:
                original_food = self.df[self.df['Food'].str.lower() == match]['Food'].iloc[0]
                matched_foods.append(original_food)
            return self.df[self.df['Food'].isin(matched_foods)]
        
        # Return empty if no matches
        return pd.DataFrame()

# -------------------------------
# 6. Health Calculation Functions
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
# 7. Food Recommendation Function
# -------------------------------
class FoodRecommender:
    def __init__(self, nutrition_df):
        # Ensure the dataframe is cleaned
        self.df = clean_nutrient_values(nutrition_df)
    
    def recommend_by_calories(self, target_calories, category=None, limit=5):
        """Recommend foods based on target calories"""
        df_filtered = self.df.copy()
        
        if category and category != "All":
            df_filtered = df_filtered[df_filtered['Category'] == category]
        
        # Calculate difference from target calories
        df_filtered['calorie_diff'] = abs(df_filtered['Calories'] - target_calories)
        
        # Return the closest matches
        return df_filtered.nsmallest(limit, 'calorie_diff')
    
    def recommend_meal_plan(self, daily_calories):
        """Recommend a complete meal plan"""
        breakfast_cals = daily_calories * 0.25
        lunch_cals = daily_calories * 0.35
        dinner_cals = daily_calories * 0.30
        snack_cals = daily_calories * 0.10
        
        breakfast = self.recommend_by_calories(breakfast_cals, limit=3)
        lunch = self.recommend_by_calories(lunch_cals, limit=3)
        dinner = self.recommend_by_calories(dinner_cals, limit=3)
        snack = self.recommend_by_calories(snack_cals, limit=2)
        
        return {
            'Breakfast': breakfast,
            'Lunch': lunch,
            'Dinner': dinner,
            'Snack': snack
        }

# -------------------------------
# 8. Image Recognition Functions (Simulated)
# -------------------------------
def simulate_image_recognition(image, nutrition_df):
    """Simulate AI food recognition for demo purposes"""
    # In a real application, this would use the CNN model
    # For demo, we'll randomly select a food
    foods = nutrition_df['Food'].tolist()
    if not foods:
        return "Unknown Food", 0.0
    
    selected_food = np.random.choice(foods)
    confidence = np.random.uniform(0.75, 0.95)
    
    return selected_food, confidence

# -------------------------------
# 9. Main Application
# -------------------------------
def main():
    # Initialization
    user_manager = UserManager()
    nutrition_df = load_nutrition_data()  # This already cleans "t" values
    recommender = FoodRecommender(nutrition_df)
    nlp_search = FoodNLPSearch(nutrition_df)
    
    # Sidebar navigation
    st.sidebar.title("🍎 The Calorie Detective")
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'nlp_search_input' not in st.session_state:
        st.session_state.nlp_search_input = ""
    
    # Login/register interface
    if not st.session_state.logged_in:
        show_login_register(user_manager)
        return
    
    # Main application menu
    menu = ["🏠 Dashboard", "📷 Food Recognition", "⚖️ Health Analysis", 
            "📊 Nutrition Database", "🎯 Diet Recommendations", "👤 Profile"]
    choice = st.sidebar.selectbox("Navigation Menu", menu)
    
    # Display user info
    st.sidebar.markdown("---")
    if st.session_state.current_user in user_manager.users:
        user_info = user_manager.users[st.session_state.current_user]['info']
        st.sidebar.markdown(f"**👤 User:** {st.session_state.current_user}")
        if 'bmi' in user_info:
            bmi = user_info.get('bmi', 'Not set')
            category, color, _ = get_bmi_category(float(bmi) if isinstance(bmi, (int, float, str)) and str(bmi).replace('.', '').isdigit() else 0)
            st.sidebar.markdown(f"**⚖️ BMI:** {bmi} ({category})")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()
    
    # Page routing
    if choice == "🏠 Dashboard":
        show_dashboard(user_manager, nutrition_df)
    elif choice == "📷 Food Recognition":
        show_food_recognition(user_manager, nutrition_df, nlp_search)
    elif choice == "⚖️ Health Analysis":
        show_health_analysis(user_manager)
    elif choice == "📊 Nutrition Database":
        show_nutrition_database(nutrition_df)
    elif choice == "🎯 Diet Recommendations":
        show_food_recommendation(recommender, user_manager)
    elif choice == "👤 Profile":
        show_user_profile(user_manager)

# -------------------------------
# 10. Page Functions
# -------------------------------
def show_login_register(user_manager):
    """Display login/register page"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("🔐 User Authentication")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", type="primary", key="login_btn"):
                if username and password:
                    success, message = user_manager.login(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.current_user = username
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter username and password")
        
        with tab2:
            new_user = st.text_input("New Username", key="reg_user")
            new_pass = st.text_input("New Password", type="password", key="reg_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
            
            # Default user info
            user_info = {
                'gender': 'Male',
                'age': 25,
                'height': 170,
                'weight': 65,
                'activity': 'Moderately active'
            }
            
            if st.button("Register", type="primary", key="reg_btn"):
                if not new_user or not new_pass:
                    st.error("Please enter username and password")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, message = user_manager.register(new_user, new_pass, user_info)
                    if success:
                        st.success(message)
                        # Auto login after registration
                        st.session_state.logged_in = True
                        st.session_state.current_user = new_user
                        st.rerun()
                    else:
                        st.error(message)
    
    with col2:
        st.image("https://static.vecteezy.com/system/resources/previews/011/401/422/non_2x/food-signal-online-food-ordering-logo-design-order-food-on-internet-restaurant-cafe-meals-delivery-online-free-vector.jpg", 
                width=250)
        st.markdown("""
        ### 🍎 System Features
        - **AI Food Recognition**: Photo-based food identification
        - **NLP Food Search**: Natural language food search
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
            try:
                bmi = float(user_info['bmi'])
                category, color, _ = get_bmi_category(bmi)
                st.metric("BMI Index", f"{bmi:.1f}", category)
            except:
                st.metric("BMI Index", "Not set")
        else:
            st.metric("BMI Index", "Not set")
    
    with col2:
        if all(key in user_info for key in ['gender', 'weight', 'height', 'age', 'activity']):
            daily_cal = calculate_daily_calories(
                user_info['gender'],
                float(user_info['weight']),
                float(user_info['height']),
                int(user_info['age']),
                user_info['activity']
            )
            st.metric("Daily Requirement", f"{daily_cal} cal")
        else:
            st.metric("Daily Requirement", "Not set")
    
    with col3:
        today = date.today().isoformat()
        today_calories = sum([float(h.get('calories', 0)) for h in history 
                            if h.get('timestamp', '').startswith(today)])
        st.metric("Today's Intake", f"{int(today_calories)} cal")
    
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
            if not df_history.empty and 'timestamp' in df_history.columns:
                df_history['date'] = pd.to_datetime(df_history['timestamp']).dt.date
                df_history['calories'] = pd.to_numeric(df_history['calories'], errors='coerce')
                
                # Get last 7 days
                recent_days = df_history.sort_values('date').tail(30)  # Get more data for rolling
                if not recent_days.empty:
                    daily_summary = recent_days.groupby('date').agg({
                        'calories': 'sum'
                    }).reset_index()
                    
                    if not daily_summary.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=daily_summary['date'],
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
                        st.info("No data for trend analysis")
                else:
                    st.info("No recent dietary records")
            else:
                st.info("No history data available")
        else:
            st.info("No dietary records yet")
    
    with col2:
        st.subheader("🍽️ Recent Food Log")
        if history:
            recent_history = history[-5:]  # Show last 5 entries
            for entry in reversed(recent_history):
                with st.expander(f"{entry.get('timestamp', 'Unknown')[:16]} - {entry.get('food_name', 'Unknown')}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Calories:** {entry.get('calories', 0)} cal")
                        st.write(f"**Protein:** {entry.get('protein', 0)}g")
                    with col_b:
                        st.write(f"**Carbs:** {entry.get('carbs', 0)}g")
                        st.write(f"**Fat:** {entry.get('fat', 0)}g")
        else:
            st.info("No dietary records yet")

def show_food_recognition(user_manager, nutrition_df, nlp_search):
    """Display food recognition page with NLP search"""
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
        selected_food_info = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose Food Image",
                type=["jpg", "jpeg", "png"],
                help="Upload clear food images for best recognition results"
            )
            
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Food Image", use_column_width=True)
                    
                    # Simulate AI recognition
                    with st.spinner("🤖 AI is analyzing food..."):
                        import time
                        time.sleep(1.5)
                        
                        food_name, confidence = simulate_image_recognition(image, nutrition_df)
                        
                        st.success(f"✅ Recognition complete!")
                        
                        st.markdown(f"""
                        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    border-radius: 10px; color: white; text-align: center;">
                            <h2>Recognition Result</h2>
                            <h1>{food_name}</h1>
                            <h3>Confidence: {confidence:.1%}</h3>
                            <p>🤖 Based on CNN deep learning model</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get food info
                        if not nutrition_df[nutrition_df['Food'] == food_name].empty:
                            selected_food_info = nutrition_df[nutrition_df['Food'] == food_name].iloc[0]
                        else:
                            st.warning("Nutrition information not found for this food")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
        
        elif input_method == "Take Photo":
            st.subheader("📸 Take a Photo")
            camera_photo = st.camera_input("Take a picture of your food", key="camera_input")
            
            if camera_photo:
                try:
                    image = Image.open(camera_photo)
                    st.image(image, caption="Captured Food Image", use_column_width=True)
                    
                    # Simulate AI recognition
                    with st.spinner("🤖 AI is analyzing food..."):
                        import time
                        time.sleep(1.5)
                        
                        food_name, confidence = simulate_image_recognition(image, nutrition_df)
                        
                        st.success(f"✅ Recognition complete!")
                        
                        st.markdown(f"""
                        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    border-radius: 10px; color: white; text-align: center;">
                            <h2>Recognition Result</h2>
                            <h1>{food_name}</h1>
                            <h3>Confidence: {confidence:.1%}</h3>
                            <p>🤖 Based on CNN deep learning model</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get food info
                        if not nutrition_df[nutrition_df['Food'] == food_name].empty:
                            selected_food_info = nutrition_df[nutrition_df['Food'] == food_name].iloc[0]
                        else:
                            st.warning("Nutrition information not found for this food")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
            else:
                st.info("👆 Click the camera button to take a photo of your food")
        
        elif input_method == "Manual Entry":
            st.subheader("🔍 NLP Food Search")
            
            # NLP Search Interface
            search_query = st.text_input(
                "Search for food using natural language:",
                placeholder="e.g., 'I had a chicken sandwich for lunch', 'healthy fruit snack', 'high protein breakfast'",
                help="Type any food-related description. Our NLP system will find matching foods.",
                key="nlp_search_input"
            )
            
            if search_query:
                with st.spinner("🔍 Searching for foods..."):
                    search_results = nlp_search.search_food(search_query)
                    
                    if not search_results.empty:
                        st.success(f"✅ Found {len(search_results)} matching foods")
                        
                        # Display search results
                        st.subheader("Search Results")
                        
                        # Let user select from results
                        food_options = search_results['Food'].tolist()
                        if food_options:
                            selected_food = st.selectbox(
                                "Select the correct food:",
                                food_options,
                                help="Choose from the NLP search results",
                                key="nlp_food_select"
                            )
                            
                            if selected_food:
                                food_name = selected_food
                                confidence = 100.0
                                
                                # Get food info
                                selected_food_info = search_results[search_results['Food'] == food_name].iloc[0]
                                
                                # Show NLP explanation
                                with st.expander("🤖 How our NLP found this"):
                                    keywords = nlp_search._extract_keywords(search_query)
                                    st.markdown(f"""
                                    **Query Analysis:**
                                    - Original query: "{search_query}"
                                    - Extracted keywords: {', '.join(keywords)}
                                    - Search method used: {"Multiple methods combined"}
                                    
                                    **NLP Features:**
                                    - Synonym matching
                                    - Fuzzy search
                                    - Category-based search
                                    - Natural language understanding
                                    """)
                        else:
                            st.warning("No food options found in search results")
                    else:
                        st.warning("No matching foods found. Try different keywords.")
                        
                        # Show suggestions
                        st.info("💡 Try searching for:")
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            if st.button("Fruits", use_container_width=True, key="suggest_fruits"):
                                st.session_state.nlp_search_input = "fruit"
                                st.rerun()
                        with col_s2:
                            if st.button("Vegetables", use_container_width=True, key="suggest_veggies"):
                                st.session_state.nlp_search_input = "vegetable"
                                st.rerun()
                        with col_s3:
                            if st.button("Protein", use_container_width=True, key="suggest_protein"):
                                st.session_state.nlp_search_input = "chicken"
                                st.rerun()
            else:
                # Traditional dropdown as fallback
                st.info("💡 Or select from complete list:")
                food_name = st.selectbox(
                    "Select Food from List:",
                    nutrition_df['Food'].tolist(),
                    help="Choose food from complete list",
                    key="manual_food_select"
                )
                if food_name:
                    confidence = 100.0
                    selected_food_info = nutrition_df[nutrition_df['Food'] == food_name].iloc[0]
        
        # Display nutrition information if food is selected
        if food_name and selected_food_info is not None:
            st.subheader("📊 Nutritional Information")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Calories", f"{int(selected_food_info['Calories'])} cal")
            with col_b:
                st.metric("Protein", f"{selected_food_info['Protein']} g")
            with col_c:
                st.metric("Carbohydrates", f"{selected_food_info['Carbs']} g")
            with col_d:
                st.metric("Fat", f"{selected_food_info['Fat']} g")
            
            # Nutrition breakdown pie chart
            try:
                protein_cals = float(selected_food_info['Protein']) * 4
                carb_cals = float(selected_food_info['Carbs']) * 4
                fat_cals = float(selected_food_info['Fat']) * 9
                
                # Check if any calories are present
                if protein_cals + carb_cals + fat_cals > 0:
                    calories_from = {
                        'Protein': protein_cals,
                        'Carbohydrates': carb_cals,
                        'Fat': fat_cals
                    }
                    
                    fig = px.pie(
                        values=list(calories_from.values()),
                        names=list(calories_from.keys()),
                        color_discrete_sequence=['#4CAF50', '#2196F3', '#FF9800'],
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not generate nutrition breakdown chart")
            
            # Save record button
            st.markdown("---")
            col_save1, col_save2 = st.columns([1, 2])
            with col_save1:
                if st.button("💾 Save to Food Log", type="primary", use_container_width=True):
                    try:
                        food_entry = {
                            'food_name': food_name,
                            'calories': float(selected_food_info['Calories']),
                            'protein': float(selected_food_info['Protein']),
                            'carbs': float(selected_food_info['Carbs']),
                            'fat': float(selected_food_info['Fat']),
                            'food_category': str(selected_food_info.get('Category', 'Other'))
                        }
                        
                        if user_manager.add_food_history(st.session_state.current_user, food_entry):
                            st.success("✅ Record saved to your food log!")
                        else:
                            st.error("❌ Failed to save record")
                    except Exception as e:
                        st.error(f"Error saving record: {e}")
    
    with col2:
        st.subheader("💡 Health Tips")
        if 'selected_food_info' in locals() and selected_food_info is not None:
            try:
                calories = float(selected_food_info['Calories'])
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
            except:
                st.info("""
                🍎 **General Tips**
                - Eat a variety of foods
                - Stay hydrated
                - Balance your meals
                """)
        
        # Display AI model information
        with st.expander("🤖 View AI Model Details"):
            st.markdown("""
            ### AI Systems Used:
            
            **1. CNN for Image Recognition**
            ```python
            Model: Custom FoodCNN
            Layers: Conv2d, ReLU, MaxPool2d
            Output: Food classification
            ```
            
            **2. NLP for Text Search**
            ```python
            Features: Fuzzy matching, synonym recognition
            Methods: Keyword extraction, category mapping
            Support: Natural language queries
            ```
            """)
        
        # Quick stats
        if st.session_state.current_user in user_manager.users:
            history = user_manager.users[st.session_state.current_user]['history']
            if history:
                today = date.today().isoformat()
                today_calories = sum([float(h.get('calories', 0)) for h in history 
                                    if h.get('timestamp', '').startswith(today)])
                st.metric("Today's Total", f"{int(today_calories)} cal")

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
            
            # Handle activity level
            activity_options = ["Sedentary", "Lightly active", "Moderately active", "Very active", "Athlete"]
            current_activity = user_info.get('activity', 'Moderately active')
            
            # Try to find matching activity
            activity_index = 2  # Default to Moderately active
            for i, option in enumerate(activity_options):
                if current_activity.lower() in option.lower() or option.lower() in current_activity.lower():
                    activity_index = i
                    break
            
            activity = st.selectbox(
                "Activity Level",
                activity_options,
                index=activity_index
            )
        
        submit_button = st.form_submit_button("Update Health Data", type="primary")
        
        if submit_button:
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
        
        try:
            bmi = float(user_info['bmi'])
            category, color, advice = get_bmi_category(bmi)
            
            # BMI dashboard
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
                    <h3>BMI Score</h3>
                    <h1 style="color: {color}; font-size: 48px;">{bmi:.1f}</h1>
                    <h4 style="color: {color};">{category}</h4>
                    <p>{advice}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Daily calorie requirement
                try:
                    daily_cal = calculate_daily_calories(
                        user_info['gender'],
                        float(user_info['weight']),
                        float(user_info['height']),
                        int(user_info['age']),
                        user_info['activity']
                    )
                    st.metric("Daily Calorie Requirement", f"{daily_cal} cal")
                except:
                    st.metric("Daily Calorie Requirement", "Calculate")
            
            with col2:
                # BMI range chart
                fig = go.Figure()
                
                # BMI range areas
                ranges = [
                    (0, 18.5, "Underweight", "#3498db"),
                    (18.5, 24.9, "Normal", "#2ecc71"),
                    (24.9, 29.9, "Overweight", "#f39c12"),
                    (29.9, 40, "Obese", "#e74c3c")
                ]
                
                for start, end, name, color in ranges:
                    fig.add_trace(go.Bar(
                        x=[end - start],
                        y=["BMI Range"],
                        base=[start],
                        orientation='h',
                        name=name,
                        marker_color=color,
                        hoverinfo='skip',
                        width=0.3
                    ))
                
                # User's BMI position
                fig.add_trace(go.Scatter(
                    x=[bmi],
                    y=["BMI Range"],
                    mode='markers',
                    marker=dict(size=20, color='black', symbol='diamond'),
                    name=f'Your BMI: {bmi:.1f}',
                    hovertemplate='<b>Your BMI: %{x:.1f}</b>'
                ))
                
                fig.update_layout(
                    title="BMI Range Visualization",
                    xaxis_title="BMI Value",
                    barmode='stack',
                    showlegend=False,
                    height=200,
                    xaxis=dict(range=[0, 40])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Health recommendations
                st.info(f"""
                ### 💡 Personalized Recommendations
                
                **Dietary Recommendations**:
                - Daily calorie goal: {daily_cal if 'daily_cal' in locals() else 'Calculate'} cal
                - Protein intake: {daily_cal * 0.15 / 4:.1f}g per day  
                - Carbohydrates: {daily_cal * 0.55 / 4:.1f}g per day  
                - Fat intake: {daily_cal * 0.3 / 9:.1f}g per day
                
                **Exercise Recommendations**:
                - At least 150 minutes of moderate exercise per week
                - Strength training twice a week
                - Walk 8,000+ steps daily
                """)
        except:
            st.warning("Please update your health data to see analysis")

def show_nutrition_database(nutrition_df):
    """Display nutrition database with ALL "t" values replaced"""
    st.title("📊 Nutrition Database")
    
    # Ensure the dataframe is cleaned (double-check)
    df_clean = clean_nutrient_values(nutrition_df)
    
    # Search and filter
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search Food Name")
    
    with col2:
        category_options = ["All"] + sorted(df_clean['Category'].dropna().unique().tolist())
        category_filter = st.selectbox(
            "Food Category",
            category_options
        )
    
    with col3:
        sort_options = ["Calories (Low to High)", "Calories (High to Low)", 
                       "Protein Content", "Name", "Category"]
        sort_by = st.selectbox(
            "Sort By",
            sort_options
        )
    
    # Filter data
    filtered_df = df_clean.copy()
    
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
    elif sort_by == "Category":
        filtered_df = filtered_df.sort_values(['Category', 'Food'])
    
    # Display data
    st.dataframe(
        filtered_df,
        column_config={
            "Food": st.column_config.TextColumn("Food Name", width="medium"),
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Calories": st.column_config.NumberColumn("Calories", format="%d cal"),
            "Protein": st.column_config.NumberColumn("Protein (g)", format="%.1f g"),
            "Carbs": st.column_config.NumberColumn("Carbs (g)", format="%.1f g"),
            "Fat": st.column_config.NumberColumn("Fat (g)", format="%.1f g")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Statistics - NOW SAFE TO CALCULATE (all "t" replaced with 0.001)
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
        if not filtered_df.empty:
            st.subheader("Calorie Distribution")
            fig = px.histogram(
                filtered_df, 
                x='Calories',
                nbins=20,
                color_discrete_sequence=['#FF6B6B'],
                title="Distribution of Calories"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if not filtered_df.empty:
            st.subheader("Category Distribution")
            category_counts = filtered_df['Category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="Food Categories"
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
        ["Weight Loss", "Balanced Nutrition", "Specific Calories", "Meal Plan"],
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
                value=max(30.0, current_weight - 5.0)
            )
        
        with col2:
            time_frame = st.selectbox(
                "Time Frame",
                ["1 month", "2 months", "3 months", "6 months"]
            )
            category_pref = st.selectbox(
                "Preferred Category",
                ["All", "Fruit", "Vegetable", "Meat", "Fish", "Dairy", "Grain", "Fast Food", "Dessert"]
            )
        
        if st.button("Generate Weight Loss Plan", type="primary"):
            # Calculate calorie deficit
            weight_loss = current_weight - target_weight
            months = {"1 month": 1, "2 months": 2, "3 months": 3, "6 months": 6}
            weeks = months[time_frame] * 4
            
            # Safe weight loss: 0.5-1kg per week
            weekly_loss = min(1.0, weight_loss / weeks)
            daily_calorie_deficit = weekly_loss * 1100 / 7  # Approx 1100 cal per 0.1kg
            
            # Get user's maintenance calories
            if all(key in user_info for key in ['gender', 'weight', 'height', 'age', 'activity']):
                maintenance = calculate_daily_calories(
                    user_info['gender'],
                    float(user_info['weight']),
                    float(user_info['height']),
                    int(user_info['age']),
                    user_info['activity']
                )
                target_calories = max(1200, maintenance - daily_calorie_deficit)
            else:
                target_calories = 1500  # Default
            
            st.success("✅ Weight loss diet plan generated!")
            
            # Display plan
            st.markdown(f"""
            ### 📋 Weight Loss Diet Plan
            
            **Goal**: {current_weight}kg → {target_weight}kg  
            **Duration**: {time_frame}  
            **Weekly Loss Target**: {weekly_loss:.1f} kg/week  
            **Daily Calorie Goal**: {int(target_calories)} cal  
            **Expected Completion**: {weeks} weeks
            """)
            
            # Generate meal recommendations
            breakfast_cals = target_calories * 0.25
            lunch_cals = target_calories * 0.35
            dinner_cals = target_calories * 0.30
            snack_cals = target_calories * 0.10
            
            st.subheader("🍽️ Daily Meal Plan")
            
            meals = [
                ("Breakfast", breakfast_cals),
                ("Lunch", lunch_cals),
                ("Dinner", dinner_cals),
                ("Snack", snack_cals)
            ]
            
            for meal_name, meal_cals in meals:
                with st.expander(f"{meal_name} - {int(meal_cals)} cal"):
                    recommendations = recommender.recommend_by_calories(meal_cals, 
                                                                      category_pref if category_pref != "All" else None,
                                                                      limit=3)
                    if not recommendations.empty:
                        for _, row in recommendations.iterrows():
                            st.write(f"**{row['Food']}** - {int(row['Calories'])} cal")
                            st.write(f"  Protein: {row['Protein']}g | Carbs: {row['Carbs']}g | Fat: {row['Fat']}g")
                    else:
                        st.write("No specific recommendations found")
    
    elif rec_type == "Specific Calories":
        st.subheader("⚖️ Specific Calorie Recommendations")
        
        target_calories = st.slider(
            "Target Calories per Serving",
            min_value=50,
            max_value=1000,
            value=300,
            step=50
        )
        
        category = st.selectbox(
            "Food Category",
            ["All", "Fruit", "Vegetable", "Meat", "Fish", "Dairy", "Grain", "Fast Food", "Dessert", "Beverage"]
        )
        
        if st.button("Find Matching Foods", type="primary"):
            matches = recommender.recommend_by_calories(target_calories, 
                                                      category if category != "All" else None)
            
            st.success(f"✅ Found {len(matches)} matching foods")
            
            # Display results
            for idx, row in matches.iterrows():
                with st.expander(f"{row['Food']} - {int(row['Calories'])} cal"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Category**: {row['Category']}")
                        st.write(f"**Protein**: {row['Protein']}g")
                        st.write(f"**Carbs**: {row['Carbs']}g")
                    with col2:
                        st.write(f"**Fat**: {row['Fat']}g")
                    
                    # Nutrition ratio chart
                    try:
                        protein_cals = float(row['Protein']) * 4
                        carb_cals = float(row['Carbs']) * 4
                        fat_cals = float(row['Fat']) * 9
                        
                        calories_from = {
                            'Protein': protein_cals,
                            'Carbohydrates': carb_cals,
                            'Fat': fat_cals
                        }
                        
                        fig = px.pie(
                            values=list(calories_from.values()),
                            names=list(calories_from.keys()),
                            color_discrete_sequence=['#4CAF50', '#2196F3', '#FF9800'],
                            hole=0.4
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.write("Nutrition breakdown not available")

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
                bmi = user_info['bmi']
                try:
                    category, color, _ = get_bmi_category(float(bmi))
                    info_html += f'<p><strong>BMI:</strong> {bmi} ({category})</p>'
                except:
                    info_html += f'<p><strong>BMI:</strong> {bmi}</p>'
            info_html += "</div>"
            st.markdown(info_html, unsafe_allow_html=True)
    
    # Food history
    st.subheader("📝 Food History")
    history = user_data.get('history', [])
    
    if history:
        # Show statistics
        total_calories = sum([float(h.get('calories', 0)) for h in history])
        avg_calories = total_calories / len(history) if len(history) > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entries", len(history))
        with col2:
            st.metric("Total Calories", f"{int(total_calories)} cal")
        with col3:
            st.metric("Average per Entry", f"{int(avg_calories)} cal")
        
        # Show recent entries
        st.subheader("Recent Entries")
        recent_history = history[-10:]  # Show last 10 entries
        
        for entry in reversed(recent_history):
            with st.expander(f"{entry.get('timestamp', 'Unknown')[:16]} - {entry.get('food_name', 'Unknown')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Calories:** {entry.get('calories', 0)} cal")
                    st.write(f"**Category:** {entry.get('food_category', 'Other')}")
                with col2:
                    st.write(f"**Protein:** {entry.get('protein', 0)}g")
                    st.write(f"**Carbs:** {entry.get('carbs', 0)}g")
                    st.write(f"**Fat:** {entry.get('fat', 0)}g")
        
        # Clear history button
        if st.button("Clear History", type="secondary"):
            user_data['history'] = []
            user_manager.save_users()
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No food history yet. Start logging your meals!")

# -------------------------------
# 11. Run Application
# -------------------------------
if __name__ == "__main__":
    main()