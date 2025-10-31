import firebase_admin
from firebase_admin import credentials, firestore

# --- path to your downloaded private key JSON ---
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# --- replace with your UID ---
uid = "VqJl2Ifr7JXNjBt9vLVIcgt3E6j2"

user_data = {
    "full_name": "Ishita Bhatia",
    "email": "ishitabhatia047@gmail.com",
    "phone": "7424981279",
    "created_at": firestore.SERVER_TIMESTAMP
}

# write into Firestore
db.collection("users").document(uid).set(user_data)

print("✅ User document created successfully!")
