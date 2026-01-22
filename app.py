"""
Hugging Face Spaces Entry Point
This file imports and runs the Streamlit app from ui.py
"""
# Fix for Streamlit + PyTorch conflict - MUST be first
import os
import sys
import warnings

os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'

warnings.filterwarnings('ignore', message='.*torch.*')
warnings.filterwarnings('ignore', message='.*RuntimeError.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)

import logging
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)

# Import the main UI module
# This will execute all the Streamlit code in ui.py
import ui

