import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
import networkx as nx
import global_variables as gv
from statistics import fmean

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

# Create NetworkX graph from data
@st.cache_data
def create_graph(data):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_node(row["source_tag_value"], weight=row["source_tag_freq"])
        G.add_node(row["target_tag_value"], weight=row["target_tag_freq"])
        G.add_edge(
            row["source_tag_value"], 
            row["target_tag_value"], 
            support=row["avg_support"]
        )
    return G, nx.arf_layout(G)

def filter_by_adj(G, node):
    neighbors = list(G.neighbors(node)) + [node]  
    return G.subgraph(neighbors)

def plot_graph(G, pos, target_node):
    node_x, node_y, node_text, node_sizes, node_colors = [], [], [], [], []
    edge_trace1, edge_trace2 = (
        go.Scatter(
            x=[], 
            y=[], 
            mode='lines', 
            line=dict(width=1), 
            hoverinfo='none'
        ), 
        go.Scatter(
            x=[], 
            y=[], 
            mode='lines', 
            line=dict(width=1), 
            hoverinfo='none')
    )

    min_size, max_size = 10, 40
    weights = {node: G.nodes[node].get('weight', 0) for node in G.nodes}
    max_weight = max(weights.values()) or 1  
    node_hovertext = [f"{name}<br>occurance: {weight}" for name, weight in weights.items()]
    
    target_neighbors = set(G.neighbors(target_node)) if target_node else set()

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        node_weight = G.nodes[node]['weight']
        normalized_size = min_size + (node_weight / max_weight) * (max_size - min_size)
        node_sizes.append(normalized_size)
        node_colors.append('orange' if node == target_node else '#8B7EC8' if node in target_neighbors else '#878580')

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace = edge_trace1 if u == target_node or v == target_node else edge_trace2
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        edge_trace['line']['width'] = 2 if edge_trace == edge_trace1 else 0.1

    edge_trace1['line']['color'], edge_trace2['line']['color'] = '#3f51b5', '#888'

    node_trace = go.Scatter(
        x=node_x, 
        y=node_y, 
        mode='markers+text',
        text=node_text,  
        textposition='top center',
        hoverinfo='text',  
        hovertext=node_hovertext, 
        marker=dict(
            color=node_colors,
            size=node_sizes,
            sizemode='diameter',
            opacity=1
        )
    )

    return go.Figure(
        data=[edge_trace1, edge_trace2, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=550 
        )
    )



if __name__ == "__main__":

    st.title("👇 Tool Co-occurrence via FP Mining")
    data = fetch_data_from_db()
    G, pos = create_graph(data)

    if "fig_state" in st.session_state:
        selected_points = st.session_state.fig_state["selection"]["points"]
        
        if len(selected_points) > 0:
            selected_choice = selected_points[0]["text"]
        else:
            selected_choice = None
    else:
        selected_choice = None


    if "filter_figure" in st.session_state:
        st.session_state.filter_figure = None
        
    filter_figure = st.sidebar.selectbox(
        label="Filter Graph On Select",
        options=[True, False],
        index = 1
    )
    
    st.sidebar.warning("❌ Distance between nodes is defined based on the layout and has no mathematical or analytical meaning.")
    
    if selected_choice:
        filtered_graph = G if not filter_figure else filter_by_adj(G, selected_choice)
    else:
        filtered_graph = G

    if filter_figure:
        pos = nx.arf_layout(filtered_graph)
    
    fig = plot_graph(filtered_graph, pos, selected_choice)

    if selected_choice:
        st.markdown(f'''
            > ### `{selected_choice.upper()}` Node Is Selected
            > 👉 Click on Node to select it
            > 👉 Click **Reset** below to get the initial network graph''')
    else:
        st.markdown(f'''
            > ### All Nodes Are Selected
            > 👉 Click on Node to select it
            > 👉 Click **Reset** to get the initial network graph''')

    st.plotly_chart(fig, use_container_width=True, height=800, on_select="rerun", key="fig_state", selection_mode="points")

    if st.button("Reset", use_container_width=True, type="primary"):
        selected_choice = None