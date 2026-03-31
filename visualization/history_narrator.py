import os
import logging
from groq import Groq

def generate_wikipedia_article(history_log: list[str], civilized_agent_name: str) -> str:
    """Uses Groq to generate a full Wikipedia-style article based on the entire history log."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "# Error: No API Key\nPlease set GROQ_API_KEY in your .env file."
        
    client = Groq(api_key=api_key)
    formatted_log = "\n".join(history_log)
    
    if not formatted_log.strip():
        formatted_log = "Nothing of note occurred over 10,000 years."

    prompt = f"""
You are an encyclopedic historian. 
Write a comprehensive Wikipedia article for a civilization guided by the {civilized_agent_name} God Agent. 
Here is the timeline of every major event that occurred in their 10,000-year history:

{formatted_log}

Your output must be formatted as beautiful Markdown. Include the following sections:
- A descriptive title (e.g., "# The Empire of the {civilized_agent_name}")
- **Introduction**: A high level summary of their legacy.
- **The Early Age**: How they started.
- **The Turning Points**: Any major wars, plagues, or climate disasters.
- **The Modern Era**: Their technological peak and final state.
- **Legacy & Morality**: Was their existence meaningful?

Write the article naturally. Do NOT just list events. Weave them into a grand historical narrative detailing their socio-political logic.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a master Wikipedia author and historian."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating Wikipedia article: {e}")
        return f"# Generator Error\nCould not reach Groq API: {e}"
