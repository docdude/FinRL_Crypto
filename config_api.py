'''
API keys loaded from .env file.
Copy .env.example to .env and fill in your keys.
'''

import os
from dotenv import load_dotenv

load_dotenv()

# Binance (original — not needed if using Alpaca)
API_KEY_BINANCE = os.getenv('API_KEY_BINANCE', 'Enter your public key here!')
API_SECRET_BINANCE = os.getenv('API_SECRET_BINANCE', 'Enter your secret key here!')

# Alpaca (free crypto data — works in all US states)
# Get keys at https://app.alpaca.markets/
# Note: Alpaca crypto historical data works even without keys (keys give higher rate limits)
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET', '')