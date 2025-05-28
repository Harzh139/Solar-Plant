import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from groq import Groq

# Load Groq API key from env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Load BLIP model & processor for image captioning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def get_image_caption(image: Image.Image) -> str:
    """Generate a caption describing the rooftop image using BLIP."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def analyze_rooftop_image(image: Image.Image) -> dict:
    """Use image caption + Groq LLM to analyze rooftop solar suitability."""
    try:
        caption = get_image_caption(image)

        prompt = f"""
You are a solar energy consultant.
Based on the rooftop image description: "{caption}", provide a report with these fields:
- Roof Type
- Usable Area (in square meters)
- Sunlight Exposure (Low, Medium, High)
- Solar Panel Recommendation (number and type)
- Installation Issues or Notes
Format your answer as:

Roof Type: ...
Usable Area: ...
Sunlight Exposure: ...
Solar Panel Recommendation: ...
Installation Issues: ...
"""

        response = client.chat.completions.create(
            model="meta-llama/llama-guard-4-12b",
            messages=[
                {"role": "system", "content": "You are a helpful solar advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        content = response.choices[0].message.content

        # Parse the response into dictionary
        results = {}
        for line in content.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                results[key.strip().lower().replace(" ", "_")] = val.strip()

        # Ensure all keys exist, fallback to N/A
        expected_keys = [
            "roof_type",
            "usable_area",
            "sunlight_exposure",
            "solar_panel_recommendation",
            "installation_issues"
        ]
        for k in expected_keys:
            if k not in results:
                results[k] = "N/A"

        return results

    except Exception as e:
        return {"error": str(e)}
