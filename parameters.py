from pathlib import Path

# Define the data folder path
DATA_FOLDER = Path(__file__).resolve().parent.parent / 'data'

# Optionally, define other commonly used paths
TRAIN_DATA_FOLDER = DATA_FOLDER / 'train'
TEST_DATA_FOLDER = DATA_FOLDER / 'test'
SUBMISSION_PATH = DATA_FOLDER / 'sample_submission.csv'
TRAIN_RLES_PATH = DATA_FOLDER / 'train_rles.csv'

# Train folder names
KIDNEY_1_DENSE = 'kidney_1_dense'
KIDNEY_1_VOL = 'kidney_1_vol'
KIDNEY_2 = 'kidney_2'
KIDNEY_3_DENSE = 'kidney_3_dense'
KIDNEY_3_SPARSE = 'kidney_3_sparse'

# Test folder names
KIDNEY_5 = 'kidney_5'
KIDNEY_6 = 'kidney_6'

