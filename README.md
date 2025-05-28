SolarScope AI 🌞

![SolarScope Demo](example_data/sample_rooftop.jpg)

A vision-powered AI tool that analyzes rooftop images for solar panel installation potential and calculates ROI.

 🚀 Project Setup



 Installation
bash
 Clone repository
git clone https://github.com/yourusername/solarscope.git
cd solarscope

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "OPENROUTER_API_KEY=your_key_here" > .env
Running the Application
bash
streamlit run app.py
🛠 Implementation Documentation
Project Structure

solar_scope/
├── app.py                     # Main Streamlit application
├── analysis.py                # Image analysis using OpenRouter AI
├── roi_calculator.py          # Financial calculations
├── utils.py                   # Helper functions
├── requirements.txt           # Dependency list
├── .env                       # API keys (gitignored)
└── example_data/              # Sample images
Key Components
Image Analysis:



Extracts roof characteristics from uploaded images



ROI Calculator:

Calculates based on Indian market rates

Default assumptions:

Panel efficiency: 20%

Cost per kW: ₹50,000

Electricity rate: ₹6/kWh

User Interface:

Streamlit-based web interface

Responsive design for mobile/desktop

📊 Example Use Cases
Residential Analysis
Upload a clear rooftop photo

Get instant analysis of usable area

View 25-year ROI projection

Commercial Assessment
Upload satellite imagery

Identify optimal panel placement

Calculate payback period

Educational Tool
Compare different roof types

Understand solar potential factors

Learn about renewable energy economics

🔮 Future Improvements
Short-Term
Add location-based sunlight data

Support for PDF building plans

Multi-roof analysis for large properties

Medium-Term
3D roof modeling integration

Government subsidy calculations

Installer recommendation engine

Long-Term
Drone imagery processing

AR preview of panels on roof

Carbon impact visualization
