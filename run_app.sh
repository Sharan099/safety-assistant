#!/bin/bash
echo "========================================"
echo "🛡️ Safety Copilot - Starting App"
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Create .env file with your API keys"
    echo ""
fi

# Run Streamlit
echo "🚀 Starting Streamlit app..."
echo ""
streamlit run streamlit_app.py

