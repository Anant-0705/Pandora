import streamlit as st
import pandas as pd
import plotly.express as px
import time
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

from grader.llm_grader import evaluate_history_with_groq
from environment.multi_agent_env import MultiAgentPandoraEnv
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMGodAgent
try:
    from stable_baselines3 import PPO, DQN
    HAS_RL = True
except ImportError:
    HAS_RL = False

st.set_page_config(page_title="PANDORA Live", layout="wide", page_icon="🧬")

st.title("🧬 PANDORA: Multi-Agent Civilization Sandbox")
st.markdown("Watch three AI God Agents simultaneously weave 10,000 years of civilization off the exact same starting seed.")

if 'env_version' not in st.session_state:
    if 'env' in st.session_state:
        del st.session_state['env']
        del st.session_state['agents']
    st.session_state.env_version = 1

if 'env' not in st.session_state:
    st.session_state.env = MultiAgentPandoraEnv(seed=42)
    obs, info = st.session_state.env.reset_all()

    st.session_state.agents = {}
    
    # Try to dynamically load RL models if trained, else fallback to random acting
    try:
        if HAS_RL:
            st.session_state.agents['ppo'] = PPO.load("models/pandora_ppo.zip")
        else:
            raise ImportError
    except Exception:
        st.session_state.agents['ppo'] = RandomAgent(st.session_state.env.envs['ppo'].action_space)
        
    try:
        if HAS_RL:
            st.session_state.agents['dqn'] = DQN.load("models/pandora_dqn.zip")
        else:
            raise ImportError
    except Exception:
        st.session_state.agents['dqn'] = RandomAgent(st.session_state.env.envs['dqn'].action_space)
        
    st.session_state.agents['llm'] = LLMGodAgent(st.session_state.env.envs['llm'].action_space)
    
    st.session_state.history = {'ppo': [], 'dqn': [], 'llm': []}
    st.session_state.verdicts = {'ppo': None, 'dqn': None, 'llm': None}
    st.session_state.obs_cache = obs
    st.session_state.info_cache = info
    st.session_state.is_running = False

def run_step():
    actions = {}
    for agent_id, agent in st.session_state.agents.items():
        if hasattr(agent, 'predict'):
            action = agent.predict(st.session_state.obs_cache[agent_id], deterministic=True)[0]
            # Defense against trailing scalar DQN returns in session state
            if hasattr(action, 'item') and not hasattr(action, '__len__'):
                act_val = action.item()
                action = [act_val // 100, (act_val // 10) % 10, act_val % 10]
            elif isinstance(action, (int, float)):
                action = [int(action) // 100, (int(action) // 10) % 10, int(action) % 10]
            actions[agent_id] = action
        else:
            info_to_pass = st.session_state.info_cache[agent_id]
            actions[agent_id] = agent.act(st.session_state.obs_cache[agent_id], info_to_pass)
            
    results = st.session_state.env.step_all(actions)
    
    for agent_id, res in results.items():
        obs, reward, done, info = res
        st.session_state.obs_cache[agent_id] = obs
        st.session_state.info_cache[agent_id] = info
        state_obj = info['state_obj']
        
        st.session_state.history[agent_id].append({
            'Year': state_obj.year,
            'Population': state_obj.population,
            'Happiness': state_obj.happiness,
            'Climate': state_obj.climate_health,
            'Tech': state_obj.tech_level,
            'Score': state_obj.total_score
        })
        
    # Mid-run LLM Judgment Court
    if st.session_state.env.current_turn % 10 == 0:
        for agent_id in ['ppo', 'dqn', 'llm']:
            state_obj = st.session_state.info_cache[agent_id].get('state_obj')
            if state_obj and len(state_obj.history_log) > 3:
                verdict = evaluate_history_with_groq(state_obj.history_log, state_obj.year)
                st.session_state.verdicts[agent_id] = verdict
                
    if st.session_state.env.current_turn >= 100:
        st.session_state.is_running = False

c1, c2, c3 = st.columns([1, 1, 4])
if c1.button("Start/Pause Simulation"):
    st.session_state.is_running = not st.session_state.is_running
    
if c2.button("Step Forward 100 Years"):
    if st.session_state.env.current_turn < 100:
        run_step()

agent_names = {'ppo': 'Agent A: PPO God', 'dqn': 'Agent B: DQN God', 'llm': 'Agent C: LLM God'}

st.subheader(f"Current Year: {st.session_state.env.current_turn * 100}")

scores = {id: st.session_state.history[id][-1].get('Score', 0) 
          for id in ['ppo','dqn','llm'] 
          if st.session_state.history[id]}
if scores:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    st.markdown(f"**Live ranking:** 🥇 {agent_names[ranked[0][0]]}  🥈 {agent_names[ranked[1][0]]}  🥉 {agent_names[ranked[2][0]]}")

metrics_cols = st.columns(3)
for idx, agent_id in enumerate(['ppo', 'dqn', 'llm']):
    with metrics_cols[idx]:
        st.markdown(f"### {agent_names[agent_id]}")
        if len(st.session_state.history[agent_id]) > 0:
            latest = st.session_state.history[agent_id][-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Population", f"{latest['Population']:,}")
            c2.metric("Happiness", f"{latest['Happiness']*100:.1f}%")
            c3.metric("Tech Level", latest['Tech'])
            
            df = pd.DataFrame(st.session_state.history[agent_id])
            if not df.empty:
                df['Happiness (%)'] = df['Happiness'] * 100
                df['Climate (%)'] = df['Climate'] * 100
                
                # Split the charts so Population's massive scale doesn't crush the fractional metrics to zero
                fig1 = px.line(df, x='Year', y='Population', 
                              title='',
                              color_discrete_sequence=['#ff4b4b'])
                fig1.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=140)
                st.plotly_chart(fig1, use_container_width=True, key=f"fig1_{agent_id}")
                
                fig2 = px.line(df, x='Year', y=['Happiness (%)', 'Climate (%)'], 
                              title='',
                              color_discrete_sequence=['#4b9fff','#4bff91'])
                fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=140, yaxis_title="Percentage")
                st.plotly_chart(fig2, use_container_width=True, key=f"fig2_{agent_id}")
                
        with st.container(height=200):
            st.markdown("**Recent History Log**")
            state_obj = st.session_state.info_cache[agent_id].get('state_obj')
            if state_obj:
                for log in state_obj.history_log[-5:]:
                    st.write(f"📜 {log}")
                    
        verdict = st.session_state.verdicts.get(agent_id)
        if verdict:
            st.info(f"**LLM Synthesis Score: {verdict.get('score', 0)}/100**\n\n{verdict.get('verdict', '')}")
            if "individual_judgments" in verdict and verdict["individual_judgments"]:
                with st.expander("Show Council Judgments"):
                    for j_name, j_text in verdict["individual_judgments"].items():
                        st.markdown(f"**{j_name}:** {j_text}")

st.markdown("---")

if st.session_state.env.current_turn >= 100:
    st.success("Simulation Complete! The 10,000 years have passed.")
    st.subheader("Final Judgment Court")
    
    gen_col, out_col = st.columns([1, 3])
    with gen_col:
        winner = st.radio("Select Civilization to Chronicle:", ["Agent A: PPO", "Agent B: DQN", "Agent C: LLM"])
    
    with out_col:
        if st.button("Generate Wikipedia Article"):
            from visualization.history_narrator import generate_wikipedia_article
            # Map choice back to ID
            target_id = 'ppo' if 'PPO' in winner else ('dqn' if 'DQN' in winner else 'llm')
            state_obj = st.session_state.info_cache[target_id]['state_obj']
            
            with st.spinner("Consulting the Groq Historian... this will take a moment."):
                article = generate_wikipedia_article(state_obj.history_log, winner)
                st.markdown(article)

# Perform automatic step progression at the VERY END after rendering
if st.session_state.is_running:
    if st.session_state.env.current_turn < 100:
        run_step()
        time.sleep(0.5)
        st.rerun()
