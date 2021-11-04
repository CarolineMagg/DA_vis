import plotly.graph_objects as go

fig_no_data_available = go.Figure()
fig_no_data_available.update_layout(
    xaxis={"visible": False},
    yaxis={"visible": False},
    annotations=[
        {
            "text": "No data available.<br>Load data",
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {
                "size": 28
            }
        }
    ]
)

fig_no_model_selected = go.Figure()
fig_no_model_selected.update_layout(
    xaxis={"visible": False},
    yaxis={"visible": False},
    annotations=[
        {
            "text": "No model(s) selected.<br>Load model(s)",
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {
                "size": 28
            }
        }
    ]
)

fig_no_data_selected = go.Figure()
fig_no_data_selected.update_layout(
    xaxis={"visible": False},
    yaxis={"visible": False},
    annotations=[
        {
            "text": "No data selected.<br>Select patient.",
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {
                "size": 28
            }
        }
    ]
)

fig_no_slice_selected = go.Figure()
fig_no_slice_selected.update_layout(
    xaxis={"visible": False},
    yaxis={"visible": False},
    annotations=[
        {
            "text": "No data selected.<br>Select slice.",
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {
                "size": 28
            }
        }
    ]
)