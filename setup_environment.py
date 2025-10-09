import logging
import os
import sys
from app.models.database import init_db
from app.models.Model_training import train_and_save_models
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("--- Starting Deployment Setup: Database & ML Models ---")

logging.info("Running database initialization (ETL)...")
if init_db():
    logging.info("Database initialized and data loaded successfully.")
else:
    logging.error("FATAL: Database initialization failed. Check logs for missing files (like the Excel data).")
    sys.exit(1)

try:
    logging.info("Running ML Model training and saving...")
    train_and_save_models()
    logging.info("ML Models trained and saved successfully into app/models/trained_models.")
except Exception as e:
    logging.error(f"FATAL: Model training and saving failed: {e}")
    sys.exit(1)

logging.info("--- Deployment Setup Complete. Ready to start Gunicorn. ---")
