import sys
import logging

sys.path.insert(0, '/cibiv/www/vhosts/alishredder')
sys.path.insert(0, '/cibiv/www/vhosts/alishredder/venv/lib/python3.6/site-packages/')

# Set up logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# Import and run the Flask app
from app import app as application
