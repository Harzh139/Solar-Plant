import streamlit as st
from PIL import Image
import re
import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from groq import Groq
from dotenv import load_dotenv
import traceback



# --- Streamlit Config (MUST BE FIRST) ---
st.set_page_config(
    page_title="SolarScope - Rooftop Analysis",
    page_icon="☀️",
    layout="centered"
)

# --- Environment Setup ---
load_dotenv()  # Load environment variables from .env file

# --- Groq API setup ---
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
    if not GROQ_API_KEY:
        st.error(" GROQ_API_KEY not found. Please set it in .env file or Streamlit secrets.")
        st.stop()
    
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    st.stop()

# --- Load BLIP model & processor ---
@st.cache_resource
def load_blip_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        return processor, model, device
    except Exception as e:
        st.error(f"Failed to load BLIP model: {str(e)}")
        st.stop()

processor, model, device = load_blip_model()

def get_image_caption(image: Image.Image) -> str:
    """Generate caption for the uploaded image using BLIP model."""
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        return processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        raise Exception(f"Image captioning failed: {str(e)}")

def analyze_rooftop_image(image: Image.Image) -> dict:
    """Analyze rooftop image using BLIP + Groq LLM."""
    try:
        caption = get_image_caption(image)

        prompt = f"""
You are a professional solar energy consultant. 
Analyze this rooftop description: "{caption}"

Provide detailed analysis with these exact fields:
- Roof Type: [flat/sloped/mixed/other]
- Usable Area: [X square meters] (estimate)
- Sunlight Exposure: [Low/Medium/High]
- Solar Panel Recommendation: [number] x [type] panels
- Installation Issues: [list any obstacles or notes]

Important Rules:
1. NEVER return "Unknown" for roof type - make your best guess
2. If area can't be determined, estimate from panel recommendation
3. For sunlight exposure, consider time of day in the image if visible
4. Be specific about panel types (monocrystalline, polycrystalline, etc.)

Format your response exactly like this example:
Roof Type: Sloped tiled roof
Usable Area: 50-60 square meters  
Sunlight Exposure: High
Solar Panel Recommendation: 12 x Monocrystalline 400W panels
Installation Issues: Small chimney on west side
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an expert solar panel installation advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        content = response.choices[0].message.content

        # Parse response into dictionary
        results = {}
        for line in content.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                clean_key = key.strip().lower().replace(" ", "_")
                results[clean_key] = val.strip()

        # Required fields with intelligent defaults
        required_fields = {
            "roof_type": "Sloped roof (estimated)",
            "usable_area": self_estimate_area(results.get("solar_panel_recommendation", "")),
            "sunlight_exposure": "Medium",
            "solar_panel_recommendation": "8 x Monocrystalline 400W panels",
            "installation_issues": "None detected"
        }
        
        for field, default in required_fields.items():
            results[field] = results.get(field, default)

        return results

    except Exception as e:
        return {"error": str(e), "details": traceback.format_exc()}

def self_estimate_area(panel_recommendation: str) -> str:
    """Estimate area when not explicitly provided"""
    try:
        # Extract number from "10 x 400W panels"
        match = re.search(r'(\d+)\s*x', panel_recommendation)
        if match:
            panel_count = int(match.group(1))
            area = panel_count * 1.7  # 1.7 sqm per panel
            return f"{area} square meters (estimated from panel count)"
    except:
        pass
    return "40 square meters (default estimate)"

def parse_usable_area(area_text: str) -> float:
    """Robust area parsing with unit conversion"""
    if "unknown" in area_text.lower():
        return None
        
    try:
        # Handle cases like "50-60 sqm" by taking average
        if "-" in area_text:
            nums = [float(n.replace(',', '')) for n in re.findall(r'[\d,.]+', area_text)]
            if nums: 
                return sum(nums) / len(nums)
        
        # Extract any number followed by area units
        match = re.search(r'([\d,.]+)\s*(sqm|m2|square meters?|平方米)', area_text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
        
        # Fallback to any standalone number
        match = re.search(r'[\d,.]+', area_text)
        if match:
            return float(match.group(0).replace(',', ''))
        
        return None
    except:
        return None

def estimate_solar_roi(usable_area_sqm: float, panel_recommendation: str = "") -> dict:
    """Realistic solar calculations with fallback estimation"""
    # Updated Constants (Indian market rates)
    PANEL_AREA = 1.7  # sqm per panel
    PANEL_WATTAGE = 400  # Watts per panel
    COST_PER_WATT = 48  # ₹ (2024 Indian rates)
    ELECTRICITY_RATE = 7  # ₹/kWh
    PRODUCTION_FACTOR = 4.5  # kWh/kW/day (India avg)
    SYSTEM_LOSSES = 0.15  # 15% system losses
    
    # Estimate area if not provided
    if usable_area_sqm is None:
        usable_area_sqm = estimate_area_from_panels(panel_recommendation)
        
    # Safety checks
    usable_area_sqm = max(min(usable_area_sqm or 40, 500), 10)  # 10-500 sqm bounds
    
    # Calculations
    panels_count = min(int(usable_area_sqm / PANEL_AREA), 100)  # Max 100 panels
    total_kw = (panels_count * PANEL_WATTAGE) / 1000
    annual_production = total_kw * PRODUCTION_FACTOR * 365 * (1 - SYSTEM_LOSSES)
    installation_cost = total_kw * 1000 * COST_PER_WATT
    annual_savings = annual_production * ELECTRICITY_RATE
    
    # ROI Calculations
    try:
        payback_years = installation_cost / annual_savings
        roi_25_years = ((annual_savings * 25) - installation_cost) / installation_cost * 100
    except ZeroDivisionError:
        payback_years = None
        roi_25_years = None
    
    return {
        "panels_count": panels_count,
        "total_kw": round(total_kw, 2),
        "annual_production_kwh": int(annual_production),
        "installation_cost_inr": int(installation_cost),
        "annual_savings_inr": int(annual_savings),
        "payback_years": round(payback_years, 1) if payback_years else None,
        "roi_25_years_percent": min(round(roi_25_years, 1), 300) if roi_25_years else None,
        "estimated_area": usable_area_sqm is None
    }

def estimate_area_from_panels(panel_recommendation: str) -> float:
    """Fallback area estimation from panel recommendation text"""
    try:
        match = re.search(r'(\d+)\s*x', panel_recommendation)
        if match:
            return int(match.group(1)) * 1.7  # 1.7 sqm per panel
    except:
        pass
    return 40  # Default average roof size

# --- Streamlit UI ---
st.title("☀️ SolarScope - Rooftop Solar Analysis")
st.markdown("Upload an image of your rooftop to get solar panel installation estimates")

with st.expander("ℹ️ How to get best results"):
    st.markdown("""
    - Use clear, high-quality images taken directly above the roof
    - Include the entire roof area in the frame
    - Avoid shadows or obstructions covering the roof
    - For sloped roofs, capture from an angle showing the surface
    """)

uploaded_file = st.file_uploader(
    "Choose a rooftop image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Rooftop", use_container_width=True)
        
        if st.button("Analyze Rooftop", type="primary"):
            with st.spinner("Analyzing your rooftop..."):
                analysis = analyze_rooftop_image(image)
                
                if "error" in analysis:
                    st.error("Analysis failed")
                    st.code(analysis["error"])
                    if "details" in analysis:
                        with st.expander("Technical details"):
                            st.text(analysis["details"])
                    st.stop()
                
                # Display analysis results
                st.subheader("🔍 Analysis Results")
                cols = st.columns(2)
                cols[0].metric("Roof Type", analysis["roof_type"])
                cols[1].metric("Sunlight Exposure", analysis["sunlight_exposure"])
                
                cols = st.columns(2)
                usable_area = parse_usable_area(analysis["usable_area"])
                
                # Show warning if area was estimated
                if usable_area is None:
                    st.warning("⚠️ Using estimated area based on panel recommendation")
                    cols[0].metric("Usable Area", analysis["usable_area"], 
                                  help="Estimated from panel recommendation")
                else:
                    cols[0].metric("Usable Area", analysis["usable_area"])
                
                cols[1].metric("Panel Recommendation", analysis["solar_panel_recommendation"])
                
                if analysis["installation_issues"].lower() not in ["none", "n/a", "none detected"]:
                    st.warning(f"⚠️ Installation Notes: {analysis['installation_issues']}")
                
                # ROI Calculation
                st.subheader("💰 Financial Estimates")
                roi_data = estimate_solar_roi(usable_area, analysis["solar_panel_recommendation"])
                
                cols = st.columns(2)
                cols[0].metric("System Size", 
                              f"{roi_data['panels_count']} panels ({roi_data['total_kw']} kW)")
                cols[1].metric("Installation Cost", 
                              f"₹{roi_data['installation_cost_inr']:,}")
                
                cols = st.columns(2)
                cols[0].metric("Annual Production", 
                              f"{roi_data['annual_production_kwh']:,} kWh")
                cols[1].metric("Annual Savings", 
                              f"₹{roi_data['annual_savings_inr']:,}")
                
                if roi_data["payback_years"]:
                    cols = st.columns(2)
                    cols[0].metric("Payback Period", 
                                 f"{roi_data['payback_years']} years",
                                 help="Time to recover installation costs through savings")
                
                if roi_data["roi_25_years_percent"]:
                    roi_percent = min(max(roi_data["roi_25_years_percent"], 0), 300)
                    progress = roi_percent / 300  # Normalize to 0-1 scale
                    
                    cols[1].metric("25-Year ROI", 
                                 f"{roi_percent}%",
                                 help="Estimated return on investment over 25 years")
                    st.progress(progress)
                    st.caption("Typical solar ROI ranges 150-300%")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()