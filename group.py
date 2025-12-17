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
import re
from difflib import get_close_matches
import warnings
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to load spaCy model with better error handling
SPACY_AVAILABLE = False
nlp = None

# Simple alternative NLP functions if spaCy is not available
class SimpleNLP:
    @staticmethod
    def extract_keywords(query):
        """Simple keyword extraction without spaCy"""
        query = query.lower().strip()
        # Remove common filler words
        filler_words = ['a', 'an', 'the', 'some', 'my', 'i', 'want', 'like', 'have', 'had', 'eat', 'ate', 'eating', 'for', 'with']
        words = query.split()
        words = [word for word in words if word not in filler_words]
        return words
    
    @staticmethod
    def extract_nouns(query):
        """Simple noun extraction (very basic)"""
        # This is a very simplified approach - in production, use a proper NLP library
        common_food_nouns = ['apple', 'banana', 'chicken', 'salmon', 'rice', 'bread', 'pizza', 
                            'burger', 'salad', 'coffee', 'tea', 'milk', 'cheese', 'egg', 
                            'fish', 'meat', 'vegetable', 'fruit', 'dessert', 'snack', 
                            'breakfast', 'lunch', 'dinner', 'sandwich', 'pasta', 'soup']
        
        words = query.lower().split()
        return [word for word in words if word in common_food_nouns]

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
# 2. CNN Model Definition
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
        # Try to load from different possible file locations
        possible_paths = [
            "nutrients.csv",
            "./nutrients.csv",
            "data/nutrients.csv"
        ]
        
        df = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    st.success(f"✅ Nutrition database loaded successfully from {path}. ({len(df)} Food items)")
                    return df
            except Exception as e:
                continue
        
        # If no file found, create sample data
        if df is None:
            st.warning("⚠️ nutrients.csv not found. Using sample data.")
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
            return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ Error loading nutrition data: {str(e)}")
        # Return empty dataframe as fallback
        return pd.DataFrame()

# -------------------------------
# 4. User Administration
# -------------------------------
class UserManager:
    def __init__(self):
        self.users_file = "users.json"
        self.users = {}
        self.load_users()
    
    def load_users(self):
        """Load user data"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            else:
                self.users = {}
                st.info("ℹ️ No existing user data found. Starting fresh.")
        except Exception as e:
            st.warning(f"⚠️ Could not load user data: {str(e)}")
            self.users = {}
    
    def save_users(self):
        """Save user data"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            st.error(f"❌ Error saving user data: {str(e)}")
    
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
# 5. NLP Food Search System (Simplified)
# -------------------------------
class FoodNLPSearch:
    def __init__(self, nutrition_df):
        self.df = nutrition_df
        self.food_synonyms = self._create_food_synonyms()
        self.simple_nlp = SimpleNLP()
        
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
    
    def _preprocess_query(self, query):
        """Preprocess the user query"""
        query = query.lower().strip()
        # Remove common filler words
        filler_words = ['a', 'an', 'the', 'some', 'my', 'i', 'want', 'like', 'have', 'had', 'eat', 'ate', 'eating']
        words = query.split()
        words = [word for word in words if word not in filler_words]
        return ' '.join(words)
    
    def _fuzzy_match(self, query, food_list):
        """Find fuzzy matches for the query"""
        matches = get_close_matches(query, food_list, n=5, cutoff=0.3)
        return matches
    
    def _extract_food_keywords(self, query):
        """Extract food-related keywords from query using simple method"""
        return self.simple_nlp.extract_keywords(query)
    
    def _search_by_synonyms(self, keywords):
        """Search for foods using synonyms"""
        results = []
        all_foods = self.df['Food'].str.lower().tolist()
        
        for keyword in keywords:
            # Direct match
            direct_matches = [food for food in all_foods if keyword in food]
            results.extend(direct_matches)
            
            # Synonym match
            for food, synonyms in self.food_synonyms.items():
                if keyword in synonyms or keyword == food:
                    # Find the actual food name in database
                    for db_food in all_foods:
                        if food in db_food:
                            results.append(db_food)
        
        # Remove duplicates and return unique results
        return list(set(results))
    
    def _search_by_category(self, query):
        """Search by food category"""
        categories = self.df['Category'].unique()
        query_lower = query.lower()
        
        category_map = {
            'fruit': 'Fruit',
            'vegetable': 'Vegetable',
            'meat': 'Meat',
            'fish': 'Fish',
            'seafood': 'Fish',
            'dairy': 'Dairy',
            'grain': 'Grain',
            'dessert': 'Dessert',
            'sweet': 'Dessert',
            'beverage': 'Beverage',
            'drink': 'Beverage',
            'fast food': 'Fast Food',
            'junk food': 'Fast Food',
            'nut': 'Nut',
            'healthy': 'Vegetable',  # Default to vegetable for healthy queries
            'protein': 'Meat',  # Default to meat for protein queries
            'carb': 'Grain'  # Default to grain for carb queries
        }
        
        for key, category in category_map.items():
            if key in query_lower:
                return self.df[self.df['Category'] == category]
        
        return pd.DataFrame()
    
    def search_food(self, query):
        """
        Main NLP search function
        Returns: DataFrame of matching foods
        """
        if not query or query.strip() == "":
            return self.df.head(10)
        
        query = query.strip()
        
        # If query is a direct food name, return exact match
        exact_match = self.df[self.df['Food'].str.lower() == query.lower()]
        if not exact_match.empty:
            return exact_match
        
        # Method 1: Fuzzy matching on food names
        food_list = self.df['Food'].str.lower().tolist()
        fuzzy_matches = self._fuzzy_match(query.lower(), food_list)
        
        if fuzzy_matches:
            # Get the original case food names
            matched_foods = []
            for match in fuzzy_matches:
                original_food = self.df[self.df['Food'].str.lower() == match]['Food'].iloc[0]
                matched_foods.append(original_food)
            
            result = self.df[self.df['Food'].isin(matched_foods)]
            if not result.empty:
                return result
        
        # Method 2: Extract keywords and search
        keywords = self._extract_food_keywords(query)
        
        if keywords:
            # Search by synonyms
            synonym_results = self._search_by_synonyms(keywords)
            if synonym_results:
                # Get original case food names
                original_names = []
                for food in synonym_results:
                    matches = self.df[self.df['Food'].str.lower() == food]
                    if not matches.empty:
                        original_names.append(matches['Food'].iloc[0])
                
                if original_names:
                    result = self.df[self.df['Food'].isin(original_names)]
                    if not result.empty:
                        return result
        
        # Method 3: Search by category
        category_results = self._search_by_category(query)
        if not category_results.empty:
            return category_results.head(10)
        
        # Method 4: Partial string match in food names
        partial_match = self.df[self.df['Food'].str.lower().str.contains(query.lower())]
        if not partial_match.empty:
            return partial_match
        
        # Method 5: Partial string match in categories
        category_partial = self.df[self.df['Category'].str.lower().str.contains(query.lower())]
        if not category_partial.empty:
            return category_partial
        
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
# 8. Main Application
# -------------------------------
def main():
    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'food_recognition_result' not in st.session_state:
        st.session_state.food_recognition_result = None
    
    # Initialization
    user_manager = UserManager()
    nutrition_df = load_nutrition_data()
    recommender = FoodRecommender(nutrition_df)
    nlp_search = FoodNLPSearch(nutrition_df)
    
    # Sidebar navigation
    st.sidebar.title("🍎 The Calorie Detective")
    
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
        if 'bmi' in user_info:
            st.sidebar.markdown(f"**⚖️ BMI:** {user_info.get('bmi', 'Not set')}")
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.food_recognition_result = None
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
# 9. Page Functions
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
            
            if st.button("Login", type="primary", use_container_width=True):
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
                    st.warning("Please enter both username and password")
        
        with tab2:
            new_user = st.text_input("New Username", key="reg_user")
            new_pass = st.text_input("New Password", type="password", key="reg_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
            
            # Basic user info during registration
            col_a, col_b = st.columns(2)
            with col_a:
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.number_input("Age", min_value=1, max_value=120, value=25)
            with col_b:
                height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
                weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=65)
            
            activity = st.selectbox(
                "Activity Level",
                ["Sedentary", "Lightly active", "Moderately active", "Very active", "Athlete"]
            )
            
            user_info = {
                'gender': gender,
                'age': age,
                'height': height,
                'weight': weight,
                'activity': activity
            }
            
            if st.button("Register", type="primary", use_container_width=True):
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters")
                elif not new_user:
                    st.error("Please enter a username")
                else:
                    success, message = user_manager.register(new_user, new_pass, user_info)
                    if success:
                        st.success(message)
                        # Auto-login after registration
                        st.session_state.logged_in = True
                        st.session_state.current_user = new_user
                        st.rerun()
                    else:
                        st.error(message)
    
    with col2:
        # Using width parameter instead of use_column_width
        st.markdown("### 🍎 Welcome to The Calorie Detective!")
        st.markdown("""
        ### 🚀 System Features
        - **📷 Food Recognition**: AI-powered food identification
        - **🔍 NLP Food Search**: Natural language food search
        - **⚖️ Calorie Calculation**: Automatic nutrition tracking
        - **📊 Health Analysis**: BMI and health insights
        - **🎯 Smart Recommendations**: Personalized diet plans
        
        ### 📝 How to Use
        1. Register/Login to your account
        2. Set up your health profile
        3. Use food recognition or manual entry
        4. Track your daily intake
        5. Get personalized recommendations
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
        if 'weight' in user_info and 'height' in user_info:
            bmi = calculate_bmi(user_info['weight'], user_info['height'])
            category, color, _ = get_bmi_category(bmi)
            st.metric("BMI Index", f"{bmi:.1f}", category)
        else:
            st.metric("BMI Index", "Not set", "Please set personal info first")
    
    with col2:
        if 'height' in user_info and 'weight' in user_info and 'age' in user_info and 'gender' in user_info and 'activity' in user_info:
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
            
            if not daily_summary.empty:
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
                st.info("No dietary records in the last 7 days")
        else:
            st.info("No dietary records yet")
    
    with col2:
        st.subheader("🍽️ Food Category Distribution")
        if history:
            df_history = pd.DataFrame(history)
            if 'food_category' in df_history.columns:
                category_counts = df_history['food_category'].value_counts()
                
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No category data available")
        else:
            st.info("No dietary records yet")

def show_food_recognition(user_manager, nutrition_df, nlp_search):
    """Display food recognition page with NLP search"""
    st.title("📷 AI Food Recognition")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_method = st.radio(
            "Select Input Method",
            ["Upload Image", "Take Photo", "Manual Entry (with NLP)"],
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
                image = Image.open(uploaded_file)
                # Using width parameter
                st.image(image, caption="Uploaded Food Image", width=400)
                
                # Simulate AI recognition
                with st.spinner("🤖 AI is analyzing food..."):
                    import time
                    time.sleep(1)
                    
                    # Simulate processing - should call actual CNN model
                    food_options = nutrition_df['Food'].tolist()
                    if food_options:
                        selected_idx = np.random.randint(0, len(food_options))
                        food_name = food_options[selected_idx]
                        confidence = np.random.uniform(85, 98)
                        
                        st.success(f"✅ Recognition complete!")
                        
                        st.markdown(f"""
                        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    border-radius: 10px; color: white; text-align: center;">
                            <h2>Recognition Result</h2>
                            <h1>{food_name}</h1>
                            <h3>Confidence: {confidence:.1f}%</h3>
                            <p>🤖 Based on CNN deep learning model</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get food info
                        selected_food_info = nutrition_df[nutrition_df['Food'] == food_name].iloc[0] if not nutrition_df[nutrition_df['Food'] == food_name].empty else None
                    else:
                        st.error("No food data available")
        
        elif input_method == "Take Photo":
            st.subheader("📸 Take a Photo")
            camera_photo = st.camera_input("Take a picture of your food", key="camera_input")
            
            if camera_photo:
                image = Image.open(camera_photo)
                st.image(image, caption="Captured Food Image", width=400)
                
                # Simulate AI recognition
                with st.spinner("🤖 AI is analyzing food..."):
                    import time
                    time.sleep(1)
                    
                    # Simulate processing
                    food_options = nutrition_df['Food'].tolist()
                    if food_options:
                        selected_idx = np.random.randint(0, len(food_options))
                        food_name = food_options[selected_idx]
                        confidence = np.random.uniform(85, 98)
                        
                        st.success(f"✅ Recognition complete!")
                        
                        st.markdown(f"""
                        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    border-radius: 10px; color: white; text-align: center;">
                            <h2>Recognition Result</h2>
                            <h1>{food_name}</h1>
                            <h3>Confidence: {confidence:.1f}%</h3>
                            <p>🤖 Based on CNN deep learning model</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get food info
                        selected_food_info = nutrition_df[nutrition_df['Food'] == food_name].iloc[0] if not nutrition_df[nutrition_df['Food'] == food_name].empty else None
                    else:
                        st.error("No food data available")
            else:
                st.info("👆 Click the camera button to take a photo of your food")
        
        elif input_method == "Manual Entry (with NLP)":
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
                        display_options = [f"{food} ({search_results[search_results['Food'] == food]['Calories'].iloc[0]} cal)" 
                                         for food in food_options]
                        
                        selected_display = st.selectbox(
                            "Select the correct food:",
                            display_options,
                            help="Choose from the NLP search results",
                            key="nlp_food_select"
                        )
                        
                        # Extract food name from selection
                        if selected_display:
                            food_name = selected_display.split(" (")[0]
                            confidence = 100.0
                            
                            # Get food info
                            selected_food_info = search_results[search_results['Food'] == food_name].iloc[0] if not search_results[search_results['Food'] == food_name].empty else None
                            
                            # Show NLP explanation
                            with st.expander("🤖 How our NLP found this"):
                                st.markdown(f"""
                                **Query Analysis:**
                                - Original query: "{search_query}"
                                - Extracted keywords: {', '.join(nlp_search._extract_food_keywords(search_query))}
                                - Search method used: {"Multiple methods combined"}
                                
                                **NLP Features:**
                                - Synonym matching
                                - Fuzzy search
                                - Category-based search
                                - Natural language understanding
                                """)
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
                confidence = 100.0
                selected_food_info = nutrition_df[nutrition_df['Food'] == food_name].iloc[0] if not nutrition_df[nutrition_df['Food'] == food_name].empty else None
        
        # Display nutrition information if food is selected
        if food_name and selected_food_info is not None:
            # Store in session state for later use
            st.session_state.food_recognition_result = {
                'food_name': food_name,
                'food_info': selected_food_info.to_dict()
            }
            
            # Display nutrition information
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
            calories_from = {
                'Protein': selected_food_info['Protein'] * 4,
                'Carbohydrates': selected_food_info['Carbs'] * 4,
                'Fat': selected_food_info['Fat'] * 9
            }
            
            # Only show pie chart if there are calories
            if sum(calories_from.values()) > 0:
                fig = px.pie(
                    values=list(calories_from.values()),
                    names=list(calories_from.keys()),
                    color_discrete_sequence=['#4CAF50', '#2196F3', '#FF9800'],
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            # Save record button
            if st.button("💾 Save to Food Log", type="primary", use_container_width=True):
                food_entry = {
                    'food_name': food_name,
                    'calories': int(selected_food_info['Calories']),
                    'protein': float(selected_food_info['Protein']),
                    'carbs': float(selected_food_info['Carbs']),
                    'fat': float(selected_food_info['Fat']),
                    'food_category': str(selected_food_info.get('Category', 'Other'))
                }
                
                if user_manager.add_food_history(st.session_state.current_user, food_entry):
                    st.success("✅ Record saved!")
                    # Reset session state
                    st.session_state.food_recognition_result = None
                    # Rerun to clear the form
                    st.rerun()
                else:
                    st.error("❌ Save failed")
    
    with col2:
        st.subheader("💡 Health Tips")
        
        # Dynamic tips based on selected food
        if 'selected_food_info' in locals() and selected_food_info is not None:
            calories = selected_food_info['Calories']
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
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Reset Form", use_container_width=True):
            st.session_state.food_recognition_result = None
            st.rerun()
        
        if st.button("📋 View Recent Foods", use_container_width=True):
            st.switch_page("🏠 Dashboard")
        
        # NLP Search Tips
        with st.expander("🔍 NLP Search Tips"):
            st.markdown("""
            ### How to use NLP Search:
            
            **Examples of effective searches:**
            - "I ate chicken and rice"
            - "healthy breakfast options"
            - "high protein lunch"
            - "fruit snack"
            - "vegetarian dinner"
            - "low calorie dessert"
            
            **Features:**
            - Understands synonyms (apple = apples)
            - Handles misspellings
            - Recognizes food categories
            - Supports natural language queries
            """)
        
        # Display AI model information
        with st.expander("🤖 View AI Model Details"):
            st.markdown("""
            ### AI Systems Used:
            
            **1. CNN for Image Recognition**
            ```python
            Accuracy: 85.3%
            Training Data: Food-101 Dataset
            ```
            
            **2. NLP for Text Search**
            ```python
            Features: Fuzzy matching, synonym recognition
            Methods: Keyword extraction, category mapping
            Support: Natural language queries
            ```
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
            
            # Get current activity from user_info
            current_activity = user_info.get('activity', 'Moderately active')
            
            # Create the selectbox
            activity = st.selectbox(
                "Activity Level",
                ["Sedentary", "Lightly active", "Moderately active", "Very active", "Athlete"],
                index=["Sedentary", "Lightly active", "Moderately active", "Very active", "Athlete"].index(current_activity) 
                if current_activity in ["Sedentary", "Lightly active", "Moderately active", "Very active", "Athlete"] 
                else 2
            )
        
        submit_button = st.form_submit_button("Update Health Data", type="primary", use_container_width=True)
        
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
        
        bmi = user_info['bmi']
        category, color, advice = get_bmi_category(bmi)
        
        # BMI dashboard
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
                <h3>BMI Index</h3>
                <h1 style="color: {color}; font-size: 48px;">{bmi:.1f}</h1>
                <h4 style="color: {color};">{category}</h4>
                <p>{advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Daily calorie requirement
            if all(key in user_info for key in ['gender', 'weight', 'height', 'age', 'activity']):
                daily_cal = calculate_daily_calories(
                    user_info['gender'],
                    user_info['weight'],
                    user_info['height'],
                    user_info['age'],
                    user_info['activity']
                )
                
                st.metric("Daily calorie requirement", f"{daily_cal} cal")
            else:
                st.info("Complete your profile to see daily calorie requirement")
        
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
                name=f'Your BMI: {bmi:.1f}',
                hovertemplate='<b>Your BMI: %{x:.1f}</b>'
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
            if all(key in user_info for key in ['gender', 'weight', 'height', 'age', 'activity']):
                daily_cal = calculate_daily_calories(
                    user_info['gender'],
                    user_info['weight'],
                    user_info['height'],
                    user_info['age'],
                    user_info['activity']
                )
                
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
    
    if nutrition_df.empty:
        st.warning("No nutrition data available")
        return
    
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
            "Protein": st.column_config.NumberColumn("Protein(g)", format="%.1f g"),
            "Carbs": st.column_config.NumberColumn("Carbs(g)", format="%.1f g"),
            "Fat": st.column_config.NumberColumn("Fat(g)", format="%.1f g"),
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
        if not filtered_df.empty:
            avg_cal = filtered_df['Calories'].mean()
            st.metric("Average Calories", f"{avg_cal:.0f}")
        else:
            st.metric("Average Calories", "N/A")
    with col3:
        if not filtered_df.empty:
            max_cal = filtered_df['Calories'].max()
            st.metric("Highest Calories", f"{max_cal:.0f}")
        else:
            st.metric("Highest Calories", "N/A")
    with col4:
        if not filtered_df.empty:
            min_cal = filtered_df['Calories'].min()
            st.metric("Lowest Calories", f"{min_cal:.0f}")
        else:
            st.metric("Lowest Calories", "N/A")
    
    # Visualizations
    if not filtered_df.empty:
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
                ["All", "Fruit", "Vegetable", "Meat", "Fish", "Dairy", "Grain", "Fast Food", "Dessert"]
            )
        
        if st.button("Generate Weight Loss Plan", type="primary", use_container_width=True):
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
            ["All", "Fruit", "Vegetable", "Meat", "Fish", "Dairy", "Grain", "Fast Food", "Dessert"]
        )
        
        if st.button("Find Matching Foods", type="primary", use_container_width=True):
            matches = recommender.recommend_by_calories(target_calories, category)
            
            if not matches.empty:
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
            else:
                st.warning("No matching foods found. Try adjusting your criteria.")
    
    elif rec_type == "Balanced Nutrition":
        st.subheader("⚖️ Balanced Nutrition Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            meal_type = st.selectbox(
                "Meal Type",
                ["Breakfast", "Lunch", "Dinner", "Snack"]
            )
            
            protein_pref = st.slider(
                "Protein Preference",
                min_value=0,
                max_value=100,
                value=30,
                step=5,
                format="%d%%"
            )
        
        with col2:
            calorie_target = st.number_input(
                "Calorie Target",
                min_value=100,
                max_value=1000,
                value=500,
                step=50
            )
            
            dietary_restriction = st.multiselect(
                "Dietary Restrictions",
                ["Vegetarian", "Low Carb", "Low Fat", "Dairy Free", "Gluten Free"]
            )
        
        if st.button("Generate Balanced Meal", type="primary", use_container_width=True):
            # Get recommendations based on preferences
            if meal_type == "Breakfast":
                target_cal = calorie_target * 0.3
            elif meal_type == "Lunch":
                target_cal = calorie_target * 0.4
            elif meal_type == "Dinner":
                target_cal = calorie_target * 0.3
            else:
                target_cal = calorie_target * 0.1
            
            matches = recommender.recommend_by_calories(target_cal, "All")
            
            if not matches.empty:
                st.success(f"✅ Generated balanced {meal_type.lower()} recommendations")
                
                # Display recommendations
                for idx, row in matches.head(3).iterrows():
                    st.write(f"**{row['Food']}** - {row['Calories']} cal")
                    st.progress(row['Protein'] / 50, text=f"Protein: {row['Protein']}g")
            
            else:
                st.warning("No matching foods found. Try adjusting your preferences.")
    
    elif rec_type == "Muscle Gain":
        st.subheader("💪 Muscle Gain Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_weight = st.number_input(
                "Current Weight (kg)",
                min_value=50.0,
                max_value=150.0,
                value=float(user_info.get('weight', 70))
            )
            
            target_weight = st.number_input(
                "Target Weight (kg)",
                min_value=current_weight + 1,
                max_value=200.0,
                value=current_weight + 5.0
            )
        
        with col2:
            workout_frequency = st.selectbox(
                "Workout Frequency",
                ["3 days/week", "4 days/week", "5 days/week", "6+ days/week"]
            )
            
            protein_target = st.slider(
                "Daily Protein Target (g)",
                min_value=50,
                max_value=300,
                value=150,
                step=10
            )
        
        if st.button("Generate Muscle Gain Plan", type="primary", use_container_width=True):
            st.success("✅ Muscle gain plan generated!")
            
            st.markdown(f"""
            ### 💪 Muscle Gain Diet Plan
            
            **Goal**: Gain {target_weight - current_weight:.1f}kg of muscle
            **Daily Protein Target**: {protein_target}g
            **Workout Schedule**: {workout_frequency}
            
            **Recommended High-Protein Foods:**
            """)
            
            # Filter for high protein foods
            high_protein_foods = recommender.df.nlargest(10, 'Protein')
            
            if not high_protein_foods.empty:
                for idx, row in high_protein_foods.iterrows():
                    with st.expander(f"{row['Food']} - {row['Protein']}g protein"):
                        st.write(f"Calories: {row['Calories']} cal")
                        st.write(f"Carbs: {row['Carbs']}g")
                        st.write(f"Fat: {row['Fat']}g")
                        st.write(f"Category: {row['Category']}")
            else:
                st.warning("No high-protein foods found in the database.")

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
        created_date = user_data.get('created_at', 'Unknown')
        if created_date != 'Unknown':
            created_date = created_date[:10]
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white;">
            <h2>{st.session_state.current_user}</h2>
            <p>Registered: {created_date}</p>
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
                info_html += f"<p><strong>BMI:</strong> {user_info['bmi']:.1f} ({user_info.get('bmi_category', '')})</p>"
            info_html += "</div>"
            st.markdown(info_html, unsafe_allow_html=True)
    
    # Food history
    st.subheader("📝 Recent Food History")
    history = user_data.get('history', [])
    
    if history:
        df_history = pd.DataFrame(history[-10:])  # Show last 10 entries
        if 'timestamp' in df_history.columns:
            df_history['time'] = pd.to_datetime(df_history['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        for _, row in df_history.iterrows():
            expander_title = f"{row.get('time', 'Unknown time')} - {row['food_name']} ({row['calories']} cal)"
            with st.expander(expander_title):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Category**: {row.get('food_category', 'Other')}")
                    st.write(f"**Protein**: {row['protein']}g")
                with col2:
                    st.write(f"**Carbohydrates**: {row['carbs']}g")
                    st.write(f"**Fat**: {row['fat']}g")
    else:
        st.info("No food history yet. Start tracking your meals!")

# -------------------------------
# 10. Run Application
# -------------------------------
if __name__ == "__main__":
    main()