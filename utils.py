import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import json

# ---------------------------------------
# FIREBASE CONFIGURATION
# ---------------------------------------
firebaseConfig = {
    "apiKey": "YOUR_API_KEY",
    "authDomain": "YOUR_PROJECT_ID.firebaseapp.com",
    "databaseURL": "",
    "projectId": "YOUR_PROJECT_ID",
    "storageBucket": "YOUR_PROJECT_ID.appspot.com",
    "messagingSenderId": "YOUR_SENDER_ID",
    "appId": "YOUR_APP_ID"
}

# Initialize Pyrebase
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Initialize Firestore
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ---------------------------------------
# USER SIGNUP AND LOGIN
# ---------------------------------------
def signup_email_password(full_name, phone, email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        user_id = user['localId']

        # Store user info in Firestore
        user_data = {
            "full_name": full_name,
            "email": email,
            "phone": phone,
            "created_at": datetime.now().isoformat()
        }
        db.collection("users").document(user_id).set(user_data)

        return True, "Signup successful! Please login."
    except Exception as e:
        error = str(e)
        if "EMAIL_EXISTS" in error:
            return False, "This email is already registered."
        else:
            return False, f"Signup error: {error}"


def login_email_password(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        user_id = user['localId']

        # Fetch user info from Firestore
        user_data = db.collection("users").document(user_id).get()
        if not user_data.exists:
            return False, None, "User data not found in Firestore."

        return True, {"id": user_id, **user_data.to_dict()}, "Login successful!"
    except Exception as e:
        return False, None, f"Login failed: {str(e)}"


# ---------------------------------------
# PORTFOLIO MANAGEMENT
# ---------------------------------------
def save_user_portfolio(user_id, portfolio_data):
    """Save or update a user's stock portfolio."""
    try:
        db.collection("portfolios").document(user_id).set({
            "portfolio": portfolio_data,
            "last_updated": datetime.now().isoformat()
        })
        return True, "Portfolio updated successfully."
    except Exception as e:
        return False, f"Error saving portfolio: {str(e)}"


def load_user_portfolio(user_id):
    """Load a user's saved portfolio."""
    try:
        doc = db.collection("portfolios").document(user_id).get()
        if doc.exists:
            data = doc.to_dict()
            return data.get("portfolio", []), data.get("last_updated", None)
        else:
            return [], None
    except Exception as e:
        return [], None


# ---------------------------------------
# FIRESTORE CLEANUP (OPTIONAL)
# ---------------------------------------
def delete_user_account(user_id):
    """Delete a user's account data (for Delete My Account)."""
    try:
        db.collection("users").document(user_id).delete()
        db.collection("portfolios").document(user_id).delete()
        return True, "Account deleted successfully."
    except Exception as e:
        return False, f"Error deleting account: {str(e)}"
