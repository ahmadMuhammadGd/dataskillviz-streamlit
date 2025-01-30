import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
import networkx as nx
import global_variables as gv
from statistics import fmean
import numpy as np 
from dataclasses import dataclass, field
from modules.GraphPlotter import PlotlyGraph

# Set Streamlit page configuration
st.set_page_config(
    page_title="FP pattern",
    page_icon="👾",
    layout="wide"
)

# Database connection function
@st.cache_data
def fetch_data_from_db():
    try:
        conn = psycopg2.connect(gv.connection_string)
        query = """
        WITH reduced_frequency_report AS (
            SELECT 
                tag_id, SUM(occurance) AS occurance
            FROM 
                warehouse.frequency_report
            GROUP BY 
                tag_id
        )
        SELECT 
            td1.tag AS source_tag_value,
            td1.tag_id AS source_tag_id,
            td2.tag AS target_tag_value,
            td2.tag_id AS target_tag_id,
            fr1.occurance AS source_tag_freq,
            fr2.occurance AS target_tag_freq,
            AVG(fp.m_support) AS avg_support
        FROM 
            warehouse.tags_fp_growth fp
        LEFT JOIN 
            warehouse.tags_dim td1 ON td1.tag_id = fp.source_tag
        LEFT JOIN 
            warehouse.tags_dim td2 ON td2.tag_id = fp.target_tag
        LEFT JOIN 
            reduced_frequency_report fr1 ON td1.tag_id = fr1.tag_id
        LEFT JOIN 
            reduced_frequency_report fr2 ON td2.tag_id = fr2.tag_id
        GROUP BY 
            td1.tag, 
            td1.tag_id, 
            td2.tag, 
            td2.tag_id,
            fr1.occurance,
            fr2.occurance;
        """
        return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    finally:
        if conn:
            conn.close()

@st.cache_data
def create_graph(data):
    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_node(row["source_tag_value"], weight=row["source_tag_freq"])
        G.add_node(row["target_tag_value"], weight=row["target_tag_freq"])
        G.add_edge(
            row["source_tag_value"], 
            row["target_tag_value"], 
            support=row["avg_support"]
        )
    return G, nx.arf_layout(G)

if __name__ == "__main__":
    st.title("👇 Tool Co-occurrence via FP Mining")
    data = fetch_data_from_db()
    network, layout = create_graph(data)
    
    if 'plotter' not in st.session_state:
        st.session_state.plotter = PlotlyGraph(network)
    plotter = st.session_state.plotter
    
    if "fig_state" in st.session_state:
        selected_points = st.session_state.fig_state["selection"]["points"]
        
        if len(selected_points) == 1:
            selected_choice = selected_points[0]["text"]
        else:
            selected_choice = None
    else:
        selected_choice = None
    
    st.sidebar.warning("❌ Distance between nodes is defined based on the layout and has no mathematical or analytical meaning.")
    
    figure = plotter.plot(network, layout, selected_choice)

    if selected_choice:
        st.sidebar.markdown(f'''
            > ### `{selected_choice.upper()}` Node Is Selected
            > 👉 Click on Node to select it            
            > 👉 Click **Reset** below to get the initial network graph''')
    else:
        st.sidebar.markdown(f'''
            > ### All Nodes Are Selected
            > 👉 Click on Node to select it               
            > 👉 Click **Reset** to get the initial network graph''')

    st.plotly_chart(figure, use_container_width=True, height=800, on_select="rerun", key="fig_state", selection_mode="points")

    if st.button("Reset", use_container_width=True, type="primary"):
        # Reset both the selection and the layout
        selected_choice = None
        st.session_state.plotter.cached_layout = None