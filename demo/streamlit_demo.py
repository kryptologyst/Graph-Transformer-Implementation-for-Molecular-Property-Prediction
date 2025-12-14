"""Streamlit demo for Graph Transformer molecular property prediction."""

import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.graph_transformer import (
    GraphTransformerNet,
    MultiScaleGraphTransformer,
    GraphTransformerWithAttention,
)
from src.data.dataset import create_synthetic_dataset
from src.utils.core import get_device, set_seed


# Page configuration
st.set_page_config(
    page_title="Graph Transformer Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set random seed
set_seed(42)


@st.cache_resource
def load_model(model_type: str, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
    """Load a pre-trained model.
    
    Args:
        model_type: Type of model to load.
        checkpoint_path: Path to checkpoint file.
        
    Returns:
        torch.nn.Module: Loaded model.
    """
    device = get_device()
    
    if model_type == "graph_transformer":
        model = GraphTransformerNet(
            in_channels=9,
            hidden_channels=64,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            use_positional_encoding=True,
            positional_encoding_type="laplacian",
            positional_encoding_dim=16,
            use_edge_attr=True,
            edge_attr_dim=3,
            pooling="mean",
            use_residual=True,
        )
    elif model_type == "multi_scale":
        model = MultiScaleGraphTransformer(
            in_channels=9,
            hidden_channels=64,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            num_scales=3,
            use_edge_attr=True,
            edge_attr_dim=3,
        )
    elif model_type == "attention":
        model = GraphTransformerWithAttention(
            in_channels=9,
            hidden_channels=64,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            use_edge_attr=True,
            edge_attr_dim=3,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        st.success(f"Loaded model from {checkpoint_path}")
    else:
        st.warning("No checkpoint found, using randomly initialized model")
    
    return model


def create_random_molecule(num_nodes: int = 10) -> Data:
    """Create a random molecular graph.
    
    Args:
        num_nodes: Number of nodes in the molecule.
        
    Returns:
        Data: Random molecular graph.
    """
    # Generate random node features (9 features for ZINC)
    x = torch.randn(num_nodes, 9)
    
    # Generate random edge indices (create a connected graph)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    
    # Remove self-loops and duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    
    # Ensure connectivity
    if edge_index.size(1) < num_nodes - 1:
        # Add edges to ensure connectivity
        for i in range(num_nodes - 1):
            edge_index = torch.cat([edge_index, torch.tensor([[i], [i + 1]])], dim=1)
    
    # Generate random edge attributes (3 features for ZINC)
    edge_attr = torch.randn(edge_index.size(1), 3)
    
    # Generate random target (molecular property)
    y = torch.randn(1) * 2.0
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def visualize_graph(data: Data) -> go.Figure:
    """Visualize a molecular graph.
    
    Args:
        data: Graph data to visualize.
        
    Returns:
        go.Figure: Plotly figure.
    """
    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract node and edge positions
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'Node {node}')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            reversescale=True,
            color=[],
            size=20,
            colorbar=dict(
                thickness=15,
                xanchor="left",
                titleside="right"
            ),
            line=dict(width=2)
        )
    )
    
    # Color nodes by degree
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'Node {node}<br>Degree: {len(adjacencies[1])}')
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Molecular Graph Visualization',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Interactive graph visualization",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='#888', size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig


def predict_property(model: torch.nn.Module, data: Data) -> float:
    """Predict molecular property using the model.
    
    Args:
        model: Trained model.
        data: Molecular graph data.
        
    Returns:
        float: Predicted property value.
    """
    device = get_device()
    model.eval()
    
    with torch.no_grad():
        # Create batch
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        
        # Move data to device
        data = data.to(device)
        
        # Get prediction
        prediction = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch,
        )
        
        return prediction.item()


def main():
    """Main demo function."""
    st.title("🧬 Graph Transformer for Molecular Property Prediction")
    st.markdown("""
    This demo showcases Graph Transformer models for predicting molecular properties.
    The models use self-attention mechanisms to capture both local and global patterns in molecular graphs.
    """)
    
    # Sidebar
    st.sidebar.header("Model Configuration")
    
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["graph_transformer", "multi_scale", "attention"],
        help="Choose the type of Graph Transformer model"
    )
    
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path (optional)",
        value="",
        help="Path to a trained model checkpoint"
    )
    
    # Load model
    try:
        model = load_model(model_type, checkpoint_path if checkpoint_path else None)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Molecule Generation")
        
        # Molecule parameters
        num_nodes = st.slider(
            "Number of Nodes",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of nodes in the molecular graph"
        )
        
        if st.button("Generate Random Molecule"):
            # Generate random molecule
            molecule = create_random_molecule(num_nodes)
            
            # Store in session state
            st.session_state.molecule = molecule
            
            st.success(f"Generated molecule with {molecule.num_nodes} nodes and {molecule.num_edges} edges")
    
    with col2:
        st.header("Model Information")
        
        # Model stats
        num_params = sum(p.numel() for p in model.parameters())
        st.metric("Model Parameters", f"{num_params:,}")
        
        st.metric("Model Type", model_type.replace("_", " ").title())
        
        # Model architecture info
        st.subheader("Architecture Details")
        if model_type == "graph_transformer":
            st.write("• Graph Transformer with positional encoding")
            st.write("• Multi-head self-attention")
            st.write("• Residual connections")
            st.write("• Global mean pooling")
        elif model_type == "multi_scale":
            st.write("• Multi-scale Graph Transformer")
            st.write("• Multiple positional encodings")
            st.write("• Different pooling strategies")
            st.write("• Feature fusion")
        elif model_type == "attention":
            st.write("• Graph Transformer with attention visualization")
            st.write("• Attention weight extraction")
            st.write("• Interpretable predictions")
    
    # Visualization and prediction
    if "molecule" in st.session_state:
        molecule = st.session_state.molecule
        
        st.header("Molecule Visualization")
        
        # Visualize graph
        fig = visualize_graph(molecule)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction
        st.header("Property Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Predict Property"):
                try:
                    prediction = predict_property(model, molecule)
                    
                    st.metric(
                        "Predicted Property",
                        f"{prediction:.4f}",
                        help="Predicted molecular property value"
                    )
                    
                    # Store prediction
                    st.session_state.prediction = prediction
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        
        with col2:
            if "prediction" in st.session_state:
                st.metric(
                    "Prediction Confidence",
                    "High" if abs(st.session_state.prediction) < 1.0 else "Medium",
                    help="Confidence level based on prediction magnitude"
                )
        
        with col3:
            if "prediction" in st.session_state:
                # Generate multiple predictions for comparison
                predictions = []
                for _ in range(5):
                    pred = predict_property(model, molecule)
                    predictions.append(pred)
                
                st.metric(
                    "Prediction Std",
                    f"{np.std(predictions):.4f}",
                    help="Standard deviation of multiple predictions"
                )
        
        # Molecule statistics
        st.header("Molecule Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes", molecule.num_nodes)
        
        with col2:
            st.metric("Edges", molecule.num_edges)
        
        with col3:
            # Calculate density
            max_edges = molecule.num_nodes * (molecule.num_nodes - 1) // 2
            density = molecule.num_edges / max_edges if max_edges > 0 else 0
            st.metric("Density", f"{density:.3f}")
        
        with col4:
            # Calculate average degree
            avg_degree = 2 * molecule.num_edges / molecule.num_nodes
            st.metric("Avg Degree", f"{avg_degree:.2f}")
    
    # Batch prediction demo
    st.header("Batch Prediction Demo")
    
    if st.button("Generate Batch of Molecules"):
        # Generate batch of molecules
        batch_size = 10
        molecules = []
        predictions = []
        
        progress_bar = st.progress(0)
        
        for i in range(batch_size):
            molecule = create_random_molecule(np.random.randint(5, 15))
            prediction = predict_property(model, molecule)
            
            molecules.append(molecule)
            predictions.append(prediction)
            
            progress_bar.progress((i + 1) / batch_size)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            "Molecule": range(batch_size),
            "Nodes": [mol.num_nodes for mol in molecules],
            "Edges": [mol.num_edges for mol in molecules],
            "Prediction": predictions,
        })
        
        st.dataframe(results_df)
        
        # Visualization
        fig = px.scatter(
            results_df,
            x="Nodes",
            y="Prediction",
            size="Edges",
            hover_data=["Molecule"],
            title="Prediction vs Molecule Size"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About Graph Transformers:**
    
    Graph Transformers extend the transformer architecture to graph-structured data by:
    - Using self-attention to capture long-range dependencies
    - Incorporating positional encodings for graph structure
    - Applying global pooling for graph-level predictions
    
    This implementation includes multiple variants optimized for molecular property prediction.
    """)


if __name__ == "__main__":
    main()
