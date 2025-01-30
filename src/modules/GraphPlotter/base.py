from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from networkx.classes.graph import Graph
import plotly.graph_objects as go
from typing import Any
import networkx as nx
import streamlit as st


class PlotGraph(ABC):
    @abstractmethod
    def plot():
        raise NotImplementedError

class PlotlyGraph(PlotGraph):
    def __init__(
        self,
        G,
        default_node_color  = '#888', 
        default_trace_color = '#888',
        target_node_color   = 'orange', 
        target_nodes_color  = '#3f51b5', 
        target_trace_color  = '#3f51b5',
        min_at_scale: int   = 10,
        max_at_scale: int   = 30
    ):
        self.G                      =   G
        self.default_node_color     =   default_node_color
        self.default_trace_color    =   default_trace_color
        self.target_node_color      =   target_node_color
        self.target_trace_color     =   target_trace_color
        self.target_nodes_color     =   target_nodes_color
        self.min_at_scale           =   min_at_scale
        self.max_at_scale           =   max_at_scale
        
        self.weights = {
            node:G.nodes[node].get('weight', 0) 
            for node in G.nodes
        }
        self.min_w   = min(self.weights.values()) or 1 
        self.max_w   = max(self.weights.values()) or 1
        self.size    = {
            node: self.min_at_scale + (weight - self.min_w) / (self.max_w - self.min_w) * (self.max_at_scale - self.min_at_scale)
            for node, weight in self.weights.items()
            }
        self.hovertext   = {
            node:f"{node} has been mentioned `{G.nodes[node].get('weight', 0):,}` times." 
            for node in G.nodes
        }
    
    def plot(self, G, layout, target: str = None) -> go.Figure:        
        pos         = layout
        node_x, node_y, node_text = [], [], []
        edge_trace1, edge_trace2 = (
            go.Scatter(
                x=[], y=[], 
                mode='lines',
                line=dict(width=2, color=self.target_trace_color),
                hoverinfo='none'
            ),
            go.Scatter(
                x=[], y=[],
                mode='lines',
                line=dict(width=0.1, color=self.target_trace_color),
                hoverinfo='none'
            )
        )
        
        annotations = []
        node_colors = []
        node_size   = []
        hover_text  = []
        
        if target:
            target_neighbors = list(G.successors(target))
        else:
            target_neighbors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            node_size.append(self.size[node])
            hover_text.append(self.hovertext[node])
            
            node_color = self.default_node_color
            if node == target:
                node_color = self.target_node_color
            elif node in target_neighbors:
                node_color = self.target_nodes_color
            node_colors.append(node_color)  
            
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            highlight = (u == target and v in target_neighbors) or (v == target and u in target_neighbors)

            if highlight:
                edge_trace = edge_trace1  
            else:
                edge_trace = edge_trace2  
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
            
            # Add arrow annotation with matching color
            annotations.append(
                dict(
                    ax=x0, ay=y0,
                    axref='x', ayref='y',
                    x=x1, y=y1,
                    xref='x', yref='y',
                    showarrow=True,
                    arrowhead=3 if highlight else 1,
                    arrowsize=10 if highlight else 1,
                    arrowwidth=0.5,
                    arrowcolor=self.target_trace_color if highlight else self.default_trace_color,
                    opacity=.5 if highlight else 0.1
                )
            )
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            hovertext = hover_text,
            marker=dict(
                color=node_colors,
                size=node_size,
                opacity=1
            )
        )
        
        fig = go.Figure(
            data=[edge_trace2, edge_trace1, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=550,
                annotations=annotations
            )
        )
        
        return fig
                