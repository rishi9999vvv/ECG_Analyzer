"""
AI Clinical Assistant Module
Uses Groq API to generate comprehensive medical-style insights
"""

import os
import json
import requests
from typing import Dict, Any, List

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')  # Use env variable
GROQ_MODEL = "llama-3.3-70b-versatile"  # Current available model - high quality
# Alternative models: "llama-3.1-8b-instant" (faster), "gemma2-9b-it" (smaller)

# Fallback: Try to get from environment variable if hardcoded key is empty
if not GROQ_API_KEY:
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

def generate_ai_insight(analysis_results: Dict[str, Any]) -> str:
    """
    Generate AI-powered clinical insights from ECG analysis results using Groq API
    
    Parameters:
    -----------
    analysis_results : dict
        ECG analysis results containing metrics and findings
    
    Returns:
    --------
    insight : str
        AI-generated clinical insight text
    """
    if not GROQ_API_KEY:
        error_msg = "⚠ ERROR: GROQ_API_KEY not set. Using fallback mode."
        print(error_msg)
        return generate_fallback_insight(analysis_results) + f"\n\n[⚠️ {error_msg}]"
    
    try:
        # Prepare comprehensive prompt from analysis results
        system_prompt = """You are an experienced cardiologist analyzing ECG data. Provide comprehensive clinical interpretation with structured analysis."""
        
        user_prompt = format_analysis_for_ai(analysis_results)
        
        print(f"🔵 Calling Groq API with key: {GROQ_API_KEY[:10]}...")
        print(f"🔵 Model: {GROQ_MODEL}")
        print(f"🔵 Prompt length: {len(user_prompt)} characters")
        
        # Prepare headers with API key
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload for Groq API (OpenAI-compatible format)
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0.6,
            "max_tokens": 500,
            "top_p": 0.9,
            "stream": False
        }
        
        # Make API call
        print("⏳ Making Groq API request...")
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Response structure: {list(result.keys())}")
            
            # Extract generated text from Groq API response
            generated_text = ""
            if 'choices' in result and len(result['choices']) > 0:
                generated_text = result['choices'][0]['message']['content']
                print(f"✅ Generated text length: {len(generated_text)} characters")
            
            if generated_text:
                print(f"✅ Groq API Success! Response received: {len(generated_text)} characters")
                # Clean and format the insight
                insight = clean_generated_text(generated_text, user_prompt)
                return insight
            else:
                error_msg = f"⚠ WARNING: Empty response from Groq API. Response: {result}"
                print(error_msg)
                return generate_fallback_insight(analysis_results) + f"\n\n[⚠️ {error_msg}]"
        else:
            error_text = response.text
            error_msg = f"❌ GROQ API ERROR: Status {response.status_code} - {error_text}"
            print(error_msg)
            return generate_fallback_insight(analysis_results) + f"\n\n[❌ {error_msg}]"
        
    except requests.exceptions.Timeout as e:
        error_msg = f"❌ GROQ API TIMEOUT: Request took longer than 30 seconds. {str(e)}"
        print(error_msg)
        return generate_fallback_insight(analysis_results) + f"\n\n[❌ {error_msg}]"
    except requests.exceptions.ConnectionError as e:
        error_msg = f"❌ GROQ API CONNECTION ERROR: Could not connect to Groq API. {str(e)}"
        print(error_msg)
        return generate_fallback_insight(analysis_results) + f"\n\n[❌ {error_msg}]"
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ GROQ API REQUEST ERROR: {str(e)}"
        print(error_msg)
        return generate_fallback_insight(analysis_results) + f"\n\n[❌ {error_msg}]"
    except KeyError as e:
        error_msg = f"❌ GROQ API RESPONSE PARSING ERROR: Missing key '{e}'. Response: {response.json() if 'response' in locals() else 'No response'}"
        print(error_msg)
        return generate_fallback_insight(analysis_results) + f"\n\n[❌ {error_msg}]"
    except Exception as e:
        error_msg = f"❌ UNEXPECTED ERROR: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return generate_fallback_insight(analysis_results) + f"\n\n[❌ {error_msg}]"


def format_analysis_for_ai(analysis_results: Dict[str, Any]) -> str:
    """
    Format analysis results into a comprehensive prompt for AI generation
    
    Parameters:
    -----------
    analysis_results : dict
        ECG analysis results
    
    Returns:
    --------
    prompt : str
        Formatted prompt for AI model
    """
    hr = analysis_results.get('avg_heart_rate', 0)
    pvc_burden = analysis_results.get('pvc_burden', 0)
    sve_burden = analysis_results.get('sve_burden', 0)
    normal_pct = analysis_results.get('normal_percentage', 100)
    hrv_metrics = analysis_results.get('hrv_metrics', {})
    sdnn = hrv_metrics.get('sdnn', 50)
    rmssd = hrv_metrics.get('rmssd', 25)
    pnn50 = hrv_metrics.get('pnn50', 0)
    total_beats = analysis_results.get('total_beats', 0)
    abnormal_beats = analysis_results.get('abnormal_beats', 0)
    
    # Build concise, action-oriented prompt
    prompt = """As a cardiologist, provide a concise ECG analysis (maximum 400 words) in this exact format:

ECG METRICS:
- Heart Rate: """ + str(hr) + """ bpm
- Normal Beats: """ + f"{normal_pct:.1f}%" + """
- PVC Burden: """ + f"{pvc_burden:.1f}%" + """
- SVE Burden: """ + f"{sve_burden:.1f}%" + """
- HRV (SDNN): """ + f"{sdnn:.1f}ms" + """ | RMSSD: """ + f"{rmssd:.1f}ms" + """ | pNN50: """ + f"{pnn50:.1f}%" + """

**ISSUES:**
[List only abnormal findings - 2-3 bullet points max]

**NEXT STEPS:**
[Immediate actions required - 2-3 bullet points max]

**TESTS TO CONSIDER:**
[Specific diagnostic tests - 2-4 bullet points max]

Keep it direct, concise, and actionable. Focus only on significant abnormalities."""
    
    return prompt


def clean_generated_text(generated_text: str, original_prompt: str) -> str:
    """
    Clean and format the generated AI text from Groq API
    
    Parameters:
    -----------
    generated_text : str
        Raw generated text from model
    original_prompt : str
        Original prompt used
    
    Returns:
    --------
    cleaned_text : str
        Cleaned and formatted insight text
    """
    # Remove the original prompt if it appears in the response
    if original_prompt in generated_text:
        insight = generated_text.replace(original_prompt, "").strip()
    else:
        insight = generated_text.strip()
    
    # Remove any leading/trailing whitespace
    insight = insight.strip()
    
    # Ensure proper formatting for structured sections
    # Format section headers (ISSUES, NEXT STEPS, TESTS, etc.)
    lines = insight.split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # Format main section headers
            if any(header in line.upper() for header in ["**ISSUES**", "**NEXT STEPS**", "**TESTS TO CONSIDER**", "ISSUES:", "NEXT STEPS:", "TESTS:", "WHAT'S WRONG", "CLINICAL INTERPRETATION", "DIAGNOSTIC TESTS"]):
                # Already has ** or add it
                if not line.startswith('**'):
                    formatted_lines.append(f"\n**{line.replace(':', '')}**")
                else:
                    formatted_lines.append(f"\n{line}")
            # Format bullet points
            elif line.startswith(('-', '•', '*')):
                formatted_lines.append(line)
            # Format numbered lists
            elif line.startswith(('1.', '2.', '3.', '4.', '5.')):
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)
    
    insight = '\n'.join(formatted_lines)
    
    # Ensure minimum length for quality
    if len(insight) < 100:
        insight = insight + "\n\n[Note: This AI-generated interpretation should be reviewed by a cardiologist for clinical decision-making.]"
    
    # Limit maximum length to keep it concise (around 400 words = ~2000 chars)
    if len(insight) > 2000:
        # Try to cut at a section boundary
        last_section = insight[:1800].rfind('\n**')
        if last_section > 1200:
            insight = insight[:last_section] + "\n\n[Analysis truncated for brevity...]"
        else:
            # Cut at last complete sentence
            insight = insight[:1800].rsplit('.', 1)[0] + "..."
    
    return insight


def generate_fallback_insight(analysis_results: Dict[str, Any]) -> str:
    """
    Generate fallback insight when AI model is not available
    
    Parameters:
    -----------
    analysis_results : dict
        ECG analysis results
    
    Returns:
    --------
    insight : str
        Fallback clinical insight
    """
    hr = analysis_results.get('avg_heart_rate', 0)
    pvc_burden = analysis_results.get('pvc_burden', 0)
    sve_burden = analysis_results.get('sve_burden', 0)
    normal_pct = analysis_results.get('normal_percentage', 100)
    hrv_metrics = analysis_results.get('hrv_metrics', {})
    sdnn = hrv_metrics.get('sdnn', 50)
    
    insights = []
    
    # Heart rate analysis
    if hr > 100:
        insights.append(f"The patient demonstrates tachycardia with a heart rate of {hr} bpm, which may indicate increased sympathetic activity, fever, or underlying cardiac conditions.")
    elif hr < 60:
        insights.append(f"The patient demonstrates bradycardia with a heart rate of {hr} bpm, which may be physiological in trained individuals or indicate conduction abnormalities.")
    else:
        insights.append(f"The patient maintains a normal heart rate of {hr} bpm, suggesting stable cardiac function.")
    
    # Arrhythmia analysis
    if pvc_burden > 10:
        insights.append(f"Significant premature ventricular contractions are present with a burden of {pvc_burden:.1f}%, which warrants clinical evaluation for underlying structural heart disease or electrolyte imbalances.")
    elif pvc_burden > 5:
        insights.append(f"Moderate PVC burden of {pvc_burden:.1f}% is observed, which may be benign in healthy individuals but requires clinical correlation.")
    
    if sve_burden > 10:
        insights.append(f"Frequent supraventricular ectopy ({sve_burden:.1f}%) may indicate atrial irritability and could predispose to atrial fibrillation.")
    
    # HRV analysis
    if sdnn < 50:
        insights.append(f"Reduced heart rate variability (SDNN: {sdnn:.1f}ms) suggests potential autonomic dysfunction and may indicate increased cardiovascular risk.")
    
    # Overall assessment
    if normal_pct > 95:
        insights.append("Overall, the rhythm appears predominantly regular with minimal ectopic activity.")
    elif normal_pct < 80:
        insights.append("The presence of frequent arrhythmias suggests the need for comprehensive cardiac evaluation.")
    
    return " ".join(insights) if insights else "The ECG analysis reveals predominantly normal findings with no significant abnormalities detected."

