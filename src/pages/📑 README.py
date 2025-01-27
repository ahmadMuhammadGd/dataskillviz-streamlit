import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
import networkx as nx
import global_variables as gv
import requests 
import json 

st.set_page_config(
    page_title="README",
    page_icon="📑",
    layout="wide"
)


# Database connection function
@st.cache_data
def fetch_data_from_db():
    conn = None
    try:
        # Connect to the database
        conn = psycopg2.connect(
            gv.connection_string
        )
        
        query_totals = """
        SELECT 
            t_lookup.title
            ,s_lookup.seniority
            ,count_per_job_title
            ,updated_at::DATE
        FROM (
            SELECT 
                job_title_id,
                seniority_id,
                updated_at,
                COUNT(DISTINCT job_id) AS count_per_job_title
            FROM 
                warehouse.tags_jobs_fact
            GROUP BY 
                job_title_id,
                seniority_id,
                updated_at
        ) as agg
        LEFT JOIN
            warehouse.titles AS t_lookup
        ON
            job_title_id = t_lookup.title_id
        LEFT JOIN 
            warehouse.seniority AS s_lookup
        ON
            s_lookup.seniority_id = agg.seniority_id
        """
        
        
        query_tools = """
        SELECT 
            tag
        FROM
            warehouse.tags_dim
        """
        
        via_query = """
        WITH ranked_data AS (
            SELECT
                via as raw_source,
                cleaned_title,
                COUNT(via) AS via_cnt,
                RANK() OVER (PARTITION BY cleaned_title ORDER BY COUNT(via) DESC) AS rnk
            FROM 
                staging.gsearch_jobs
            GROUP BY
                via,
                cleaned_title
        )
        SELECT
            raw_source,
            cleaned_title,
            via_cnt
        FROM
            ranked_data
        WHERE
            rnk <= 5
        ORDER BY
            cleaned_title,
            rnk;
        """
        
        return (
          pd.read_sql(query_totals, conn), 
          pd.read_sql(query_tools, conn),
          pd.read_sql(via_query, conn),
          )
    except Exception as e:
      raise Exception(e)
      


if __name__ == "__main__":
    st.markdown("# README")
    data_totals, data_tools, data_via = fetch_data_from_db()
    
    batch_report = data_totals.groupby(['title', 'updated_at']).sum('count_per_job_title').sort_values(by='updated_at', ascending=True)
    batch_report = batch_report.rename(columns={
        'title': 'job title',
        'updated_at': 'ingested at',
        'count_per_job_title': 'ingested records count'
    })

    total_jobs = data_totals['count_per_job_title'].sum()
    total_tools = data_tools.count()

    # Display metrics
    st.markdown("> Before starting, we should review the data context and biases to validate this analysis.")
    col1, col2 = st.columns(2)
    col1.metric(label="Total Collected Jobs", value=f"{total_jobs:,} jobs")
    col2.metric(label="Total Collected Tools", value=total_tools)

    st.divider()
    st.info("""
    👉 The **Streamlit UI**, **Vector Embeddings**, and **Data Warehouse** were developed by [Ahmad Muhammad](https://www.linkedin.com/in/ahmadmuhammadgd). The data was originally collected by [**Luke Barousse**](https://www.lukebarousse.com) and uploaded to [Kaggle](https://www.kaggle.com/datasets/lukebarousse/data-analyst-job-postings-google-search).
    👉 If you encounter any issues, feel free to leave me a message.
    """)

    st.markdown("## Data Distribution")

    col1, col2 = st.columns(2)
    col1.markdown("### Job Title And Seniority Distribution")
    col1.markdown("""
        Using Regex, it is possible to classify job titles and seniority based on predefined rules. **this method is not flawless** """)
    col1.bar_chart(
      horizontal=True,
      data = data_totals,
      x = 'title',
      y = 'count_per_job_title',
      color = 'seniority',
      stack=False,
      height=400
    )

    col2.markdown("### Job Posting Sources and Job Titles")
    col2.markdown("The job source (via) column is raw and has not been cleaned, as it is not relevant to my area of interest in this project.")
    col2.bar_chart(
      horizontal=True,
      data = data_via,
      x = 'raw_source',
      y = 'via_cnt',
      color='cleaned_title',
      height=400
    )
    
    
    st.markdown("## ETL Reports")
    st.dataframe(batch_report, use_container_width=True)
    
    
    tools_to_show = json.loads(requests.get('https://raw.githubusercontent.com/ahmadMuhammadGd/skillVector-assets/refs/heads/main/target_keywords.json').content)
    st.markdown('## Collected Tools')
    st.write(tools_to_show)