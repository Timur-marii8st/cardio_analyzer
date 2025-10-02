"""
Script to create an admin user
Usage: python -m tools.create_admin
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from apps.api.services.storage import Storage, DbConfig
from apps.api.auth import AuthService, UserCreate
from apps.api.settings import api_settings

def main():
    print("Creating admin user...")
    
    storage = Storage(DbConfig(url=api_settings.database_url))
    auth_service = AuthService(storage)
    
    admin_user = UserCreate(
        email="admin@example.com",
        password="cardio_anal",
        full_name="System Administrator",
        role="admin",
        department="IT"
    )
    
    try:
        user = auth_service.register_user(admin_user)
        print(f"✓ Admin user created successfully!")
        print(f"  Email: {user.email}")
        print(f"  User ID: {user.user_id}")
        print(f"  Role: {user.role}")
        print("\n⚠️  Please change the default password!")
    except Exception as e:
        print(f"✗ Failed to create admin user: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()