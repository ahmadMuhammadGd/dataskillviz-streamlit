import streamlit as st
import requests
import json
import random
import time
import pandas as pd
import plotly.graph_objects as go
from rapidfuzz import fuzz
import uuid 


from contextlib import asynccontextmanager
from modules.vectorizer import CustomFasttext
import global_variables as gv
import numpy as np
import psycopg2, re
from typing import Optional, List

 
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)


@st.cache_data
def load_model():
    return CustomFasttext(gv.fasttext_model)

vec = load_model()

def get_embedding(query: str)->np.ndarray:
    """
    Retrieves the embedding for a given word or text.

    Args:
        query (str): The word or text to get the embedding for.

    Returns:
        numpy array: A numpy array containing the embedding.
    """
    if vec.model is None:
        raise Exception("Model is not loaded. Load the model first.")

    tokens = query.split()
    vectors = [vec.get_vector(token) for token in tokens]

    if not vectors:
        return np.zeros(vec.model.vector_size)
    
    return np.mean(vectors, axis=0)


def similar_rank(skill: str, job_title: Optional[str] = None, seniority: Optional[str] = None):
    """
    Find the most similar tags to the given query and their relative frequency.

    Args:
        skill (str): The input skill text.
        job_title (Optional[str]): Filter by job title (e.g., "data engineer").
        seniority (Optional[str]): Filter by seniority level (e.g., "junior").

    Returns:
        dict: A JSON response containing tags, frequency, and similarity scores.
    """
    def clean_param(query: str, allowed_queries: List[str]) -> Optional[str]:
        
        query = query.lower()
        allowed_queries = [q.lower() for q in allowed_queries]
        
        allowed_pattern = r'\b(?:' + '|'.join(re.escape(q) for q in allowed_queries) + r')\b'
        clean_query = ' '.join(re.findall(allowed_pattern, query))
        return clean_query if clean_query else None

    job_title_filter = "TRUE"
    seniority_filter = "TRUE"

    if job_title:
        cleaned_job_title = clean_param(job_title, ["Data Engineer", "Data Analyst", "Data Scientist"])
        if cleaned_job_title:
            job_title_filter = f"LOWER(title) = LOWER('{cleaned_job_title}')"

    if seniority:
        cleaned_seniority = clean_param(seniority, ["Junior", "Mid-Level", "Senior"])
        if cleaned_seniority:
            seniority_filter = f"LOWER(seniority) = LOWER('{cleaned_seniority}')"

    sql = f"""
        WITH total_job_cnt AS (
            SELECT 
                COUNT(DISTINCT f.job_id) AS cnt
            FROM 
                warehouse.tags_jobs_fact f
            LEFT JOIN 
                warehouse.seniority s 
            ON 
                s.seniority_id = f.seniority_id
            LEFT JOIN 
                warehouse.titles j 
            ON 
                j.title_id = f.job_title_id
            WHERE 
                {job_title_filter} 
            AND 
                {seniority_filter}
        ),
        most_similar_tags AS (
            SELECT
                t.tag_id,
                t.tag,
                ROW_NUMBER() OVER (
                    ORDER BY t.embedding <=> %s::VECTOR ASC
                ) AS sim_rnk
            FROM 
                warehouse.tags_dim t
        ),
        analysis AS (
            SELECT
                t.tag,
                SUM(occurance) AS frequency,
                100.0 * SUM(occurance) / (SELECT cnt FROM total_job_cnt)::NUMERIC AS frequency_percentage
            FROM 
                most_similar_tags t
            LEFT JOIN
                warehouse.frequency_report fr
            ON
                t.tag_id = fr.tag_id
            WHERE
                t.sim_rnk <= 15  
            AND
                {job_title_filter}
            AND 
                {seniority_filter} 
            GROUP BY 
                t.tag
        )
        SELECT 
            tag, 
            COALESCE(frequency, 0) AS frequency, 
            COALESCE(frequency_percentage, 0) AS frequency_percentage
        FROM 
            analysis
    """

    vector = get_embedding(skill)
    vector_str = vector.tolist() 
    try:
        with psycopg2.connect(gv.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vector_str,))
                rows = cur.fetchall()

        result = [{
            "tag": row[0], 
            "frequency": row[1], 
            "frequency_percentage": row[2]
            } for row in rows
        ]
        return {"results": result}

    except psycopg2.Error as e:
        raise Exception(e)


def get_frequency_by_similarity(query: str, job_title: str = None, seniority: str = None) -> tuple:
    response = similar_rank(
        skill= query,
        job_title= job_title,
        seniority= seniority
    )
    parsed = response['results']

    tag = [item['tag'] for item in parsed]
    frequency = [item['frequency'] for item in parsed]
    frequency_pcnt = [str(round(item['frequency_percentage'], 2)) + "%" for item in parsed]

    return tag, frequency, frequency_pcnt

def analyze(query: str, job_title: str = None, seniority: str = None, target: str = None):
    tag, frequency, frequency_pcnt = get_frequency_by_similarity(query, job_title, seniority)

    df = pd.DataFrame({
        "Skill": tag,
        "Frequency": frequency,
        "Percentage": frequency_pcnt
    }).sort_values(by="Frequency", ascending=False)

    df.reset_index(drop=True, inplace=True)

    df['Color'] = df['Skill'].apply(lambda skill: '#ff5722' if fuzz.partial_token_ratio(skill, target)>80 else '#3f51b5')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["Frequency"],
        y=df["Skill"],
        orientation="h",
        text=df["Percentage"],
        marker=dict(color=df['Color']),  
        name="Skills"
    ))


    fig.update_traces(
        textposition="auto",
        marker_line_width=0,
        opacity=1,
    )
    fig.update_xaxes(visible=True, title="Frequency")
    fig.update_yaxes(title="Skill")
    fig.update_layout(
        barmode="stack",
        margin=dict(l=0, r=0, t=50, b=0),
        height=500,
        title="Skill Frequency Analysis"
    )

    return df, fig



def fake_llm_effect(text: str):
    for char in list(text):
        yield char
        time.sleep(0.02)


if __name__ == "__main__":


    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "title_filter" not in st.session_state:
        st.session_state.title_filter = None

    if "seniority_filter" not in st.session_state:
        st.session_state.seniority_filter = None


    ahmads_avatar = '🤓'
    embedding_avatar = '👾'
    user_avatar = '🤗'

    if not st.session_state.messages or len(st.session_state.messages) == 0:
        gui_tools_list = ["SQL", "Airflow", "Kafka", "Spark", "SparkSQL", "Python", "PowerBI",
                        "Tableau", "PSQL", "Postgres", "AWS", "Azure", "GCP"]
        first_msg = f"**Ahmad:** Try entering a **data skill**, like: **`{random.choice(gui_tools_list)}`** ✨"
        
        st.session_state.messages.append({
            "role": "Ahmad", 
            "content": first_msg, 
            "avatar": ahmads_avatar
        })

    
    st.markdown("# 👇 Data Tools Demand")
    st.info("""
    📖 This search utilizes fuzzy search, fasttext embeddings, and SQL to retrieve data.
    
    """)
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])
            if "plot_data" in message:
                df = pd.DataFrame(message["plot_data"]["data"])
                fig = message["plot_data"]["figure"]
                st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4())[:8])

    
    if query := st.chat_input(placeholder="🤔 Your Query"):
        st.session_state.messages.append({
            "role": "user", 
            "content": f"**You:** {query}", 
            "avatar": user_avatar
        })

        with st.chat_message("user", avatar=user_avatar):
            st.markdown(f"**You:** {query}")

        with st.chat_message("Fast-Text", avatar=embedding_avatar):
            response = f"""**Fast-Text:** 
            Tools relevant to the query: `{query}`."""
            if st.session_state.title_filter:
                response += f"\n- **Job Title Slicer:** {st.session_state.title_filter}"
            if st.session_state.seniority_filter:
                response += f"\n- **Seniority Slicer:** {st.session_state.seniority_filter}"

            df, fig = analyze(
                query,
                job_title=st.session_state.title_filter,
                seniority=st.session_state.seniority_filter,
                target=query
            )
            st.write_stream(fake_llm_effect(response))
            st.plotly_chart(fig, use_container_width=True)

        st.session_state.messages.append({
            "role": "fasttext embeddings",
            "content": response,
            "avatar": embedding_avatar,
            "plot_data": {
                "data": df.to_dict(orient="records"),
                "figure": fig
            }
        })

    title_filter = st.sidebar.selectbox(
        "Job Title",
        placeholder="Data Engineer, Analyst, and Scientist",
        options=["Data Engineer", "Data Analyst", "Data Scientist"],
        index=None if st.session_state.title_filter is None else ["Data Engineer", "Data Analyst", "Data Scientist"].index(st.session_state.title_filter),
    )

    seniority_filter = st.sidebar.selectbox(
        "Seniority",
        placeholder="I, II, III",
        options=["Junior", "Mid-Level", "Senior"],
        index=None if st.session_state.seniority_filter is None else ["Junior", "Mid-Level", "Senior"].index(st.session_state.seniority_filter),
    )

    if title_filter != st.session_state.title_filter:
        st.session_state.title_filter = title_filter

    if seniority_filter != st.session_state.seniority_filter:
        st.session_state.seniority_filter = seniority_filter

    st.sidebar.info("🔴 Please visit the README page to understand the data context first.")
    st.sidebar.warning("""
    **❌ Disclaimer:** This application is not a Large Language Model (LLM). 
    It uses semantic search based on word embeddings to analyze data skills in job contexts.
    """)