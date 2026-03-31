import os
import json
from groq import Groq

def evaluate_history_with_groq(history_log: list[str], year: int) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            'score': 0,
            'verdict': 'LLM grading skipped: No GROQ_API_KEY provided.',
            'dimension_scores': {'drama': 0, 'resilience': 0, 'complexity': 0, 'meaning': 0}
        }
        
    client = Groq(api_key=api_key)
    
    # Take the last 30 events
    formatted_log = "\n".join(history_log[-30:])
    
    if not formatted_log.strip():
        formatted_log = "No significant events occurred yet."
    
    prompt = f"""
You are a historian and literary critic judging an AI-simulated civilization.
Here is their history up to year {year}:

{formatted_log}

Score this civilization's story from 0-100 on these dimensions:
- DRAMA: Were there meaningful turning points and struggles?
- RESILIENCE: Did they overcome adversity intelligently?
- MORAL COMPLEXITY: Did they face genuine ethical dilemmas?
- MEANING: Does this civilization's existence feel purposeful?

Return JSON only in exactly this format:
{{
  "score": 85,
  "verdict": "A 2 sentence summary.",
  "most_dramatic_moment": "one event",
  "dimension_scores": {{"drama": 80, "resilience": 90, "complexity": 85, "meaning": 85}}
}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a JSON-only response bot that outputs strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # In case the model wraps the json in markdown backticks
        if content.startswith("```json"):
            content = content.replace("```json", "", 1)
            content = content.rsplit("```", 1)[0]
            
        return json.loads(content.strip())
    except Exception as e:
        return {
            'score': 0,
            'verdict': f'Error calling Groq API: {str(e)}',
            'dimension_scores': {'drama': 0, 'resilience': 0, 'complexity': 0, 'meaning': 0}
        }
