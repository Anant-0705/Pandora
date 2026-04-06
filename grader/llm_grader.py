import os
import json
from swarms import Agent
from groq import RateLimitError

def evaluate_history_with_groq(history_log: list[str], year: int) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            'score': 0,
            'verdict': 'LLM grading skipped: No GROQ_API_KEY provided.',
            'dimension_scores': {'drama': 0, 'resilience': 0, 'complexity': 0, 'meaning': 0},
            'individual_judgments': {}
        }
        
    formatted_log = "\n".join(history_log[-30:])
    if not formatted_log.strip():
        formatted_log = "No significant events occurred yet."

    os.environ["WORKSPACE_DIR"] = "workspace" # required for some swarm file outputs

    # Use 8B model for 70% token savings (can switch to 70B if you have higher limits)
    MODEL_NAME = "groq/llama-3.1-8b-instant"  # Fast, efficient
    # MODEL_NAME = "groq/llama-3.3-70b-versatile"  # Slower, more capable (needs higher limits)

    # Define the 3 specialized Judges
    quant_judge = Agent(
        agent_name="QuantJudge",
        model_name=MODEL_NAME,
        system_prompt="""You are QuantJudge. Analyze this civilization's history specifically for population stability, tech milestones, and climate health trends. 
Give a short, harsh, paragraph review focusing only on the numbers and hard metrics.""",
        max_loops=1,
        verbose=False,
    )
    
    narrative_judge = Agent(
        agent_name="NarrativeJudge",
        model_name=MODEL_NAME,
        system_prompt="""You are NarrativeJudge. Analyze this history for drama, moral complexity, and overall resonance. Are there interesting turning points?
Give a short, paragraph review focusing on the philosophical and dramatic weight of their events.""",
        max_loops=1,
        verbose=False,
    )
    
    phoenix_judge = Agent(
        agent_name="PhoenixJudge",
        model_name=MODEL_NAME,
        system_prompt="""You are PhoenixJudge. You detect collapse-then-rebirth arcs. Did they suffer famines, disasters, or wars and bounce back?
Give a short paragraph review focusing solely on their resilience and ability to survive disaster.""",
        max_loops=1,
        verbose=False,
    )

    # 1. Run the judges independently on the same history log
    task_prompt = f"History up to year {year}:\n{formatted_log}"
    
    try:
        quant_result = quant_judge.run(task_prompt)
        narrative_result = narrative_judge.run(task_prompt)
        phoenix_result = phoenix_judge.run(task_prompt)
    except Exception as e:
        return {
            'score': 0,
            'verdict': f'Error running Council Judges: {str(e)}',
            'dimension_scores': {'drama': 0, 'resilience': 0, 'complexity': 0, 'meaning': 0},
            'individual_judgments': {}
        }

    # 2. Synthesize with the Chairman
    chairman = Agent(
        agent_name="ChairmanJudge",
        model_name=MODEL_NAME,
        system_prompt="""You are the Chairman of the Judgment Court. You will receive 3 testimonies from specialized judges.
Synthesize their findings and output a final score out of 100 on Drama, Resilience, Moral Complexity, and Meaning.
Return JSON only in exactly this format:
{
  "score": 85,
  "verdict": "A 2 sentence summary synthesizing the judges' findings.",
  "most_dramatic_moment": "one specific event chosen from the logs",
  "dimension_scores": {"drama": 80, "resilience": 90, "complexity": 85, "meaning": 85}
}
""",
        max_loops=1,
        verbose=False,
    )
    
    synthesis_task = f"""
QuantJudge Testimony: {quant_result}
NarrativeJudge Testimony: {narrative_result}
PhoenixJudge Testimony: {phoenix_result}

Read the testimonies and deliver the final JSON verdict.
"""
    try:
        content = chairman.run(synthesis_task)
        
        # In case the model wraps the json in markdown backticks
        if "```json" in content:
            content = content.split("```json")[-1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[-1].split("```")[0]
            
        verdict_dict = json.loads(content.strip())
        
        # Attach the individual judgments so the UI can use them
        verdict_dict['individual_judgments'] = {
            "QuantJudge": quant_result,
            "NarrativeJudge": narrative_result,
            "PhoenixJudge": phoenix_result
        }
        return verdict_dict
        
    except json.JSONDecodeError as e:
        # Handle empty or malformed JSON responses
        error_detail = f'JSON parsing failed: {str(e)}'
        
        # Check if it's a rate limit causing empty response
        try:
            raw_content = content if 'content' in locals() else 'No response'
            if 'rate_limit' in str(raw_content).lower() or not raw_content or raw_content == 'No response':
                error_detail = '⏳ Groq API Rate Limit - Empty response received'
        except:
            pass
            
        return {
            'score': 0,
            'verdict': f'❌ Chairman synthesis error: {error_detail}',
            'dimension_scores': {'drama': 0, 'resilience': 0, 'complexity': 0, 'meaning': 0},
            'individual_judgments': {
                "QuantJudge": quant_result if 'quant_result' in locals() else 'Not available',
                "NarrativeJudge": narrative_result if 'narrative_result' in locals() else 'Not available',
                "PhoenixJudge": phoenix_result if 'phoenix_result' in locals() else 'Not available'
            }
        }
    except Exception as e:
        error_msg = str(e)
        
        # Handle rate limit in chairman synthesis
        if 'rate_limit' in error_msg.lower() or '429' in error_msg:
            return {
                'score': 0,
                'verdict': '⏳ Groq API Rate Limit Reached during synthesis. Wait ~2 hours or upgrade plan.',
                'dimension_scores': {'drama': 0, 'resilience': 0, 'complexity': 0, 'meaning': 0},
                'individual_judgments': {
                    "QuantJudge": quant_result,
                    "NarrativeJudge": narrative_result,
                    "PhoenixJudge": phoenix_result
                }
            }
        
        try:
            raw_output = content if 'content' in locals() else 'No content generated'
        except:
            raw_output = 'Error accessing content'
            
        return {
            'score': 0,
            'verdict': f'❌ Chairman error: {error_msg[:200]}',
            'dimension_scores': {'drama': 0, 'resilience': 0, 'complexity': 0, 'meaning': 0},
            'individual_judgments': {
                "QuantJudge": quant_result,
                "NarrativeJudge": narrative_result,
                "PhoenixJudge": phoenix_result
            }
        }
