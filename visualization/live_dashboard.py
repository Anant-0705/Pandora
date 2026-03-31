import streamlit as st
import pandas as pd
import plotly.express as px
import time
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

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
    st.session_state.obs_cache = obs
    st.session_state.info_cache = info
    st.session_state.is_running = False

def run_step():
    actions = {}
    for agent_id, agent in st.session_state.agents.items():
        if hasattr(agent, 'predict'):
            actions[agent_id] = agent.predict(st.session_state.obs_cache[agent_id], deterministic=True)[0]
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
            'Tech': state_obj.tech_level
        })
        
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
                # Plot multiple axis data by shifting scales, or just show population
                fig = px.line(df, x='Year', y='Population', title='Population Tracker', color_discrete_sequence=['#ff4b4b'])
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250)
                st.plotly_chart(fig, use_container_width=True)
                
        with st.expander("Recent History Log", expanded=False):
            state_obj = st.session_state.info_cache[agent_id].get('state_obj')
            if state_obj:
                for log in state_obj.history_log[-5:]:
                    st.write(f"📜 {log}")

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
