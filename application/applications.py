########################################################################################################################
# Visualization application with DASH+Plotly
########################################################################################################################

import ast
import json
import cv2
import pandas as pd
import numpy as np
import dash
from dash import dcc, Output, Input, State
from dash import html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

from application.default_figures import *
from data_utils.DataContainer import DataContainer
from data_utils.TestSet import TestSet

app = dash.Dash(__name__)
app.config['suppress_callback_exceptions'] = True

MAX_VALUE_ASSD = 362
MODELS_SIMPLE1 = ["XNet_T2_relu", "XNet_T2_leaky", "XNet_T2_selu"]
MODELS_SIMPLE2 = ["XNet_T1_relu", "XNet_T1_leaky", "XNet_T1_selu"]
MODELS_SIMPLE_CG = ["CG_XNet_T1_relu", "CG_XNet_T2_relu"]
MODELS_BASELINE = [*MODELS_SIMPLE1, *MODELS_SIMPLE2, *MODELS_SIMPLE_CG]
MODELS_SEGMS2T = ["SegmS2T_GAN1_relu", "SegmS2T_GAN2_relu", "SegmS2T_GAN5_relu",
                  "CG_SegmS2T_GAN1_relu", "CG_SegmS2T_GAN2_relu", "CG_SegmS2T_GAN5_relu"]
MODELS_GAN_XNET = ["GAN_1+XNet_T1_relu", "GAN_2+XNet_T1_relu", "GAN_5+XNet_T1_relu",
                   "GAN_1+CG_XNet_T1_relu", "GAN_2+CG_XNet_T1_relu", "GAN_5+CG_XNet_T1_relu"]
MODELS_DA = [*MODELS_SEGMS2T, *MODELS_GAN_XNET]
MODELS = [*MODELS_BASELINE, *MODELS_DA]
MODELS_CG = [*MODELS_SIMPLE_CG,
             "CG_SegmS2T_GAN1_relu", "CG_SegmS2T_GAN2_relu", "CG_SegmS2T_GAN5_relu",
             "GAN_1+CG_XNet_T1_relu", "GAN_2+CG_XNet_T1_relu", "GAN_5+CG_XNet_T1_relu"]
MODELS_NOT_CG = [*MODELS_SIMPLE1, *MODELS_SIMPLE2,
                 "SegmS2T_GAN1_relu", "SegmS2T_GAN2_relu", "SegmS2T_GAN5_relu",
                 "GAN_1+XNet_T1_relu", "GAN_2+XNet_T1_relu", "GAN_5+XNet_T1_relu"]

METRICS = ["DSC", "ASSD", "ACC", "TPR", "TNR"]

# header
div_load_data = html.Div(
    id="div_load_data",
    style={'width': '33%',
           'backgroundColor': 'grey'},
    children=[html.Div(children=[html.Button('Load Data', id='load_data')],
                       style={'width': '20%', 'display': 'inline-block'}),
              html.Div(children=[dcc.Loading(id="loading_data",
                                             type="default",
                                             children=html.Div(id="loading_data_output")
                                             )],
                       style={'width': '30%', 'display': 'inline-block', 'textAlign': 'left'}),
              dcc.Store(id='df_total'),
              ])

# control panel
div_overview_metric = html.Div(
    children=[html.Div(children=html.Label('Error Metric:'),
                       style={"width": '20%',
                              "textAlign": "left",
                              'display': 'table-cell'}),
              html.Div(children=dcc.Dropdown(id='dropdown_overview_metric',
                                             options=[
                                                 {'label': 'DSC (DiceSimilarityCoefficient)',
                                                  'value': 'DSC'},
                                                 {'label': 'ASSD (AverageSymmetricSurfaceDistance',
                                                  'value': 'ASSD'},
                                                 {'label': 'ACC (Accuracy)',
                                                  'value': 'ACC'},
                                                 {'label': 'TPR (TruePositiveRate)',
                                                  'value': 'TPR'},
                                                 {'label': 'TNR (TrueNegativeRate)',
                                                  'value': 'TNR'}
                                             ],
                                             value='DSC'
                                             ),
                       style={"width": '80%',
                              'display': 'table-cell'})
              ],
    style={'display': 'table',
           'width': '97%'})
div_overview_dataset = html.Div(
    children=[html.Div(children=html.Label('Dataset:'),
                       style={'width': '20%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=dcc.RadioItems(id="radioitem_overview_slices",
                                               options=[
                                                   {'label': 'only tumor',
                                                    'value': 'only_tumor'},
                                                   {'label': 'all',
                                                    'value': 'all'}
                                               ],
                                               value='only_tumor'
                                               ),
                       style={'width': '80%',
                              'textAlign': 'left',
                              'display': 'table-cell'})
              ],
    style={'display': 'table',
           'width': '97%'})
div_overview_model = html.Div(
    children=[html.Div(children=html.Label("Model:"),
                       style={'width': '20%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=dcc.Dropdown(id="dropdown_overview_model",
                                             options=[{"label": "All", "value": "All"}] +
                                                     [{"label": "Baseline", "value": "Baseline"}] +
                                                     [{"label": "DA", "value": "DA"}] +
                                                     [{"label": "CG", "value": "CG"}] +
                                                     [{"label": "NOT_CG", "value": "NOT_CG"}] +
                                                     [{"label": "SegmS2T", "value": "SegmS2T"}] +
                                                     [{"label": "Gen+Segm", "value": "Gen+Segm"}] +
                                                     [{"label": k, "value": k} for k in MODELS],
                                             placeholder="Select a list of models...",
                                             value=["DA"],
                                             multi=True),
                       style={'width': '70%',
                              'display': 'table-cell',
                              'font-size': '85%'}),
              html.Div(children=html.Button(id='submit_overview', n_clicks=0, children='Show'),
                       style={'width': '10%',
                              'display': 'table-cell',
                              'align': 'center'})
              ],
    style={'display': 'table',
           'width': '97%'})
div_overview_control_panel = html.Div(
    children=[html.H2('All Patients', id="header_overview", style={"textAlign": "center"}),
              div_overview_metric,
              html.Br(),
              div_overview_dataset,
              html.Br(),
              div_overview_model,
              dcc.Store(id="df_metric_overview")],
    style={'display': 'table-cell',
           'width': '32%'})
div_detail_metric = html.Div(
    children=[html.Div(children=html.Label('Error Metric:'),
                       style={"width": '20%',
                              "textAlign": "left",
                              'display': 'table-cell'}),
              html.Div(children=dcc.Dropdown(id='dropdown_detail_metric',
                                             options=[
                                                 {'label': 'DSC (DiceSimilarityCoefficient)',
                                                  'value': 'DSC'},
                                                 {'label': 'ASSD (AverageSymmetricSurfaceDistance',
                                                  'value': 'ASSD'},
                                                 {'label': 'ACC (Accuracy)',
                                                  'value': 'ACC'},
                                                 {'label': 'TPR (TruePositiveRate)',
                                                  'value': 'TPR'},
                                                 {'label': 'TNR (TrueNegativeRate)',
                                                  'value': 'TNR'}
                                             ],
                                             value='DSC'
                                             ),
                       style={"width": '80%',
                              'display': 'table-cell'})
              ],
    style={'display': 'table',
           'width': '97%'})
div_detail_dataset = html.Div(
    children=[html.Div(children=html.Label('Dataset:'),
                       style={'width': '20%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=dcc.RadioItems(id="radioitem_detail_slices",
                                               options=[
                                                   {'label': 'only tumor',
                                                    'value': 'only_tumor'},
                                                   {'label': 'all',
                                                    'value': 'all'}
                                               ],
                                               value='only_tumor'
                                               ),
                       style={'width': '80%',
                              'textAlign': 'left',
                              'display': 'table-cell'})
              ],
    style={'display': 'table',
           'width': '97%'})
div_detail_model = html.Div(
    children=[html.Div(children=html.Label("Model:"),
                       style={'width': '20%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=dcc.Dropdown(id="dropdown_detail_model",
                                             options=[{"label": "All", "value": "All"}] +
                                                     [{"label": "Baseline", "value": "Baseline"}] +
                                                     [{"label": "DA", "value": "DA"}] +
                                                     [{"label": "CG", "value": "CG"}] +
                                                     [{"label": "NOT_CG", "value": "NOT_CG"}] +
                                                     [{"label": "SegmS2T", "value": "SegmS2T"}] +
                                                     [{"label": "Gen+Segm", "value": "Gen+Segm"}] +
                                                     [{"label": k, "value": k} for k in MODELS],
                                             placeholder="Select a list of models...",
                                             value=["DA"],
                                             multi=True),
                       style={'width': '70%',
                              'display': 'table-cell',
                              'font-size': '85%'}),
              html.Div(children=html.Button(id='submit_detail', n_clicks=0, children='Show'),
                       style={'width': '10%',
                              'display': 'table-cell'})
              ],
    style={'display': 'table',
           'width': '100%'})
div_detail_control_panel = html.Div(
    children=[html.H2('Patient', id="header_detail", style={"textAlign": "center"}),
              div_detail_metric,
              html.Br(),
              div_detail_dataset,
              html.Br(),
              div_detail_model,
              dcc.Store(id="df_metric_detail"),
              dcc.Store(id="df_patient_id")
              ],
    style={'display': 'table-cell',
           'width': '32%'})
div_visual_view = html.Div(
    children=[html.Div(children=html.Label('2D View:'),
                       style={'width': '20%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=dcc.RadioItems(id="radioitem_visual_view",
                                               options=[
                                                   {'label': 'single',
                                                    'value': 'single'},
                                                   {'label': 'multiple',
                                                    'value': 'multiple'}
                                               ],
                                               value='single'
                                               ),
                       style={'width': '80%',
                              'textAlign': 'left',
                              'display': 'table-cell'})
              ],
    style={'display': 'table',
           'width': '97%'})
switches_model_single = html.Div(
    children=[html.Div(children=html.Label("Masks:"),
                       style={'width': '20%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=dcc.Checklist(id="switches_mask_single",
                                              options=[{"label": "GT", "value": 1}, {"label": "Pred", "value": 2}],
                                              value=[1, 2],
                                              inputStyle={"margin-right": "5px"},
                                              labelStyle={'display': 'inline-block'}),
                       style={'width': '80%',
                              'padding-left': '5px',
                              'textAlign': 'left',
                              'display': 'table-cell'})],
    style={'display': 'table',
           'width': '97%'})
switches_model_multi = html.Div(
    children=[html.Div(children=dcc.Checklist(id="switches_mask_multi",
                                              options=[],
                                              value=[],
                                              labelStyle={'display': 'inline-block'}),
                       style={'width': '100%',
                              'textAlign': 'left',
                              'display': 'table-cell'})],
    style={'display': 'table',
           'width': '97%'})
div_visual_control_panel = html.Div(
    children=[html.H2('Slice', style={"textAlign": "center"}),
              div_visual_view,
              switches_model_single,
              switches_model_multi],
    style={"display": 'table-cell',
           'width': '32%'}
)
div_control_panel = html.Div(
    id="control_panel",
    style={'width': '100%',
           'display': 'table',
           'backgroundColor': 'lightgray'},
    children=[div_overview_control_panel,
              div_detail_control_panel,
              div_visual_control_panel,
              dcc.Store(id="df_2d_data"),
              dcc.Store(id="df_3d_data"),
              dcc.Store(id="df_patient_data")])

# heatmaps
div_overview_heatmap = html.Div(
    children=[dcc.Graph(id="overview_plot",
                        figure=fig_no_data_available),
              dcc.RangeSlider(id='overview_plot_slider',
                              min=0,
                              max=1,
                              step=0.01,
                              value=[0, 1],
                              tooltip={"placement": "bottom",
                                       "always_visible": True},
                              allowCross=False)],
    style={'display': 'table-cell',
           'width': '33.3%'}
)
div_detail_heatmap = html.Div(
    children=[dcc.Graph(id="detail_plot", figure=fig_no_data_selected),
              dcc.RangeSlider(id='detail_plot_slider',
                              min=0,
                              max=1,
                              step=0.01,
                              value=[0, 1],
                              tooltip={"placement": "bottom",
                                       "always_visible": True},
                              allowCross=False)],
    style={'display': 'table-cell',
           'width': '33.3%'}
)
div_visual_2d = html.Div(
    children=[html.Div(children=[dcc.Slider(id="visual_slice_slider",
                                            min=0,
                                            max=80,
                                            step=1,
                                            value=0,
                                            tooltip={"always_visible": True},
                                            vertical=True)],
                       style={"width": "8%", "height": "96%",
                              "display": "inline-block", "position": "relative", "margin-bottom": "5%"}),
              html.Div(children=[dcc.Slider(id='visual_model_slider',
                                            min=0,
                                            max=len(MODELS) + 1,
                                            step=None,
                                            value=0,
                                            vertical=False),
                                 dcc.Graph(id="visual_plot", figure=fig_no_slice_selected)],
                       style={"width": "92%", "height": "96%",
                              "display": "inline-block", "position": "relative"}),
              ],
    style={'display': 'table-cell',
           'width': '33.3%'}
)
div_heatmaps = html.Div(
    id="heatmaps",
    style={'width': '100%',
           'display': 'table',
           'backgroundColor': 'gray'},
    children=[div_overview_heatmap,
              div_detail_heatmap,
              div_visual_2d]
)

# layout
app.layout = html.Div(
    children=[
        html.H1(children='Hello Dash',
                style={'textAlign': 'center',
                       'width': '100%'}),
        div_load_data,
        div_control_panel,
        div_heatmaps
    ])


def get_colorscale_tickvals(metric, slider_values, slider_max):
    # define colorscale and tickvals
    lookup_color = list(reversed([*px.colors.sequential.Plasma])) if metric == "ASSD" else [
        *px.colors.sequential.Plasma]
    steps = (slider_values[1] - slider_values[0]) / 9
    colorscale = []
    if slider_values[0] != 0:
        colorscale.append([0, lookup_color[0]])
    for idx, x in enumerate(np.arange(slider_values[0], slider_values[1], steps)):
        colorscale.append([x / slider_max, lookup_color[idx]])
    colorscale.append([slider_values[1] / slider_max, lookup_color[-1]])
    if slider_values[1] != 1:
        colorscale.append([1, lookup_color[-1]])
    tickvals = np.arange(0, slider_max, 20) if metric == "ASSD" else np.arange(0, 1, 0.1)
    return colorscale, tickvals


def get_selected_model_list(models, fixed):
    models_selected = None
    if "All" not in models:
        models_selected = fixed + models
        if "Baseline" in models:
            models_selected += [m for m in MODELS_BASELINE]
            models_selected.remove("Baseline")
        if "DA" in models:
            models_selected += [m for m in MODELS_DA]
            models_selected.remove("DA")
        if "CG" in models:
            models_selected += [m for m in MODELS_CG]
            models_selected.remove("CG")
        if "NOT_CG" in models:
            models_selected += [m for m in MODELS_NOT_CG]
            models_selected.remove("NOT_CG")
        if "SegmS2T" in models:
            models_selected += [m for m in MODELS_SEGMS2T]
            models_selected.remove("SegmS2T")
        if "Gen+Segm" in models:
            models_selected += [m for m in MODELS_GAN_XNET]
            models_selected.remove("Gen+Segm")
        seen = set()
        models_selected = [x for x in models_selected if not (x in seen or seen.add(x))]
    return models_selected


@app.callback(
    Output("loading_data_output", "children"),
    Output("df_total", "data"),
    Input("load_data", "n_clicks"))
def load_data_spinner(n_clicks):
    # if n_clicks == 0 or n_clicks is None:
    #     raise PreventUpdate
    # else:
    testset = TestSet("/tf/workdir/data/VS_segm/VS_registered/test_processed/", load=True,
                      data_load=False, evaluation_load=False)
    df_total = testset.df_total
    return "Data is loaded.", df_total.to_json()


@app.callback(
    Output("overview_plot_slider", "step"),
    Output("overview_plot_slider", "max"),
    Output("overview_plot_slider", "value"),
    Input("dropdown_overview_metric", "value"))
def update_slider_overview(metric):
    steps = {"DSC": 0.01,
             "ASSD": 1,
             "ACC": 0.01,
             "TPR": 0.01,
             "TNR": 0.01}
    max_values = {"DSC": 1,
                  "ASSD": MAX_VALUE_ASSD,
                  "ACC": 1,
                  "TPR": 1,
                  "TNR": 1}
    return steps[metric], max_values[metric], [0, max_values[metric]]


@app.callback(
    Output("detail_plot_slider", "step"),
    Output("detail_plot_slider", "max"),
    Output("detail_plot_slider", "value"),
    Input("dropdown_detail_metric", "value"))
def update_slider_detail(metric):
    steps = {"DSC": 0.01,
             "ASSD": 1,
             "ACC": 0.01,
             "TPR": 0.01,
             "TNR": 0.01}
    max_values = {"DSC": 1,
                  "ASSD": MAX_VALUE_ASSD,
                  "ACC": 1,
                  "TPR": 1,
                  "TNR": 1}
    return steps[metric], max_values[metric], [0, max_values[metric]]


@app.callback(
    Output("visual_slice_slider", "value"),
    Output("visual_slice_slider", 'min'),
    Output("visual_slice_slider", "max"),
    Output("visual_slice_slider", "marks"),
    Input("df_metric_detail", "data"),
    Input("df_patient_data", "data")
)
def update_slice_slider_2d(info, info2):
    if info is None or info2 is None:
        raise PreventUpdate
    else:
        df = pd.read_json(info)
        df2 = json.loads(info2)
        if len(df) == 0 or len(df2) == 0:
            raise PreventUpdate
        min_slice = df.iloc[0]["slice"]
        max_slice = df.iloc[-2]["slice"]
        current_slice = df2["slice_id"]
        if max_slice - min_slice < 20:
            marks = {k: {'label': str(k)} for k in range(min_slice, max_slice + 1, 1)}
        else:
            marks = {k: {'label': str(k)} if idx % 5 == 0 else {'label': ""} for idx, k in
                     enumerate(range(min_slice, max_slice + 1, 1))}
            marks[max_slice] = {'label': str(max_slice)}
        return current_slice, min_slice, max_slice, marks


@app.callback(
    Output("visual_model_slider", "value"),
    Output("visual_model_slider", "max"),
    Input("dropdown_detail_model", "value"),
    Input("df_patient_data", "data")
)
def update_model_slider_2d(list_models, info2):
    if list_models is None or info2 is None:
        raise PreventUpdate
    else:
        df2 = json.loads(info2)
        if len(list_models) == 0 or len(df2) == 0:
            raise PreventUpdate
        max_model = len(list_models) - 1
        current_model = list_models.index(df2["model"])
        print("current model slicer", current_model)
        return current_model, max_model


# @app.callback(
#     Output('click-data', 'children'),
#     Input('overview_plot', 'clickData'))
# def display_click_data(clickData):
#     return json.dumps(clickData, indent=2)

@app.callback(
    Output('click-data', 'children'),
    Input('overview_plot', 'relayoutData'))
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)


@app.callback(
    Output("df_metric_overview", "data"),
    Output("dropdown_overview_model", "value"),
    Input("df_total", "data"),
    Input("submit_overview", "n_clicks"),
    Input("dropdown_overview_metric", "value"),
    State("dropdown_overview_model", "value"),
    Input("radioitem_overview_slices", "value"))
def update_data_overview(info_total, n_clicks, metric, models, slice_type):
    if info_total is None:
        raise PreventUpdate
    elif models is None or len(models) == 0:
        return pd.DataFrame().to_json(), []
    else:
        df_total = pd.read_json(info_total)
        lookup = {"DSC": f"dice_{slice_type}",
                  "ASSD": f"assd_{slice_type}",
                  "ACC": f"acc_{slice_type}",
                  "TPR": f"tpr_{slice_type}",
                  "TNR": f"tnr_{slice_type}"}
        df_metric = pd.DataFrame(df_total.iloc[0][lookup[metric]])
        models_selected = get_selected_model_list(models, fixed=["id"])
        if models_selected is not None:
            df_metric = df_metric[models_selected]
        models_selected = list(df_metric.columns)[1:]
        return df_metric.to_json(), models_selected


@app.callback(
    Output("overview_plot", "figure"),
    Input("df_metric_overview", "data"),
    Input("overview_plot_slider", "value"),
    Input("overview_plot_slider", "max"))
def update_heatmap_overview(info, slider_values, slider_max):
    if info is None:
        raise PreventUpdate
    else:
        # read dataframe
        df_metric = pd.read_json(info)
        if len(df_metric) == 0:
            return fig_no_model_selected
        metric = df_metric.iloc[-1]["id"]

        # define colorscale and tickvals
        colorscale, tickvals = get_colorscale_tickvals(metric, slider_values, slider_max)

        # create figure
        fig = make_subplots(rows=2, cols=1,
                            row_heights=[0.1, 0.9], vertical_spacing=0.05, shared_xaxes=True)
        # create annotated heatmap with total values
        round_dec = 2 if len(df_metric.columns) >= 8 else 3
        trace = ff.create_annotated_heatmap(x=list(df_metric.columns)[1:],
                                            y=["mean"],
                                            z=[list(df_metric.iloc[-1][1:].values)],
                                            hoverinfo='skip',
                                            coloraxis="coloraxis",
                                            annotation_text=[
                                                [np.round(x, round_dec) for x in list(df_metric.iloc[-1][1:].values)]])
        fig.add_trace(trace.data[0],
                      1, 1)
        fig.layout.update(trace.layout)

        # prepare x,y,z for heatmap
        x = list(df_metric.columns)[1:]
        y = list(df_metric["id"].values[:-1])
        z = [list(df_metric.iloc[idx][1:].values) for idx in range(len(df_metric) - 1)]
        # create hovertext
        hovertext = list()
        for yi, yy in enumerate(y):
            hovertext.append(list())
            for xi, xx in enumerate(x):
                hovertext[-1].append(
                    'Model: {}<br />ID: {}<br />{}: {}'.format(xx, yy, metric, np.round(z[yi][xi], decimals=5)))
        # heatmap for patient data
        fig.add_trace(go.Heatmap(x=x,
                                 y=y,
                                 z=z,
                                 hoverongaps=True,
                                 hoverinfo='text',
                                 text=hovertext,
                                 coloraxis="coloraxis"), 2, 1)
        # update layout
        fig.update_layout(xaxis2={'showticklabels': False},
                          xaxis1={'side': 'top', 'showticklabels': True},
                          yaxis2={'title': 'Patient ID'},
                          coloraxis={'colorscale': colorscale,
                                     'colorbar': dict(title=metric, tickvals=tickvals, tickmode="array")},
                          margin=dict(l=5,
                                      r=5,
                                      b=5,
                                      t=5,
                                      pad=4)
                          )
    return fig


@app.callback(
    Output("df_patient_id", "data"),
    Output("header_detail", "children"),
    Input("overview_plot", "clickData"))
def update_patient_id_detail(clickData):
    if clickData is None:
        raise PreventUpdate
    else:
        patient_id = clickData["points"][0]["y"]
        return json.dumps(patient_id), f"Patient ID {patient_id}"


@app.callback(
    Output("df_patient_data", "data"),
    Output("switches_mask_multi", "options"),
    Input("detail_plot", "clickData"),
    Input("dropdown_detail_model", "value"),
    Input("radioitem_visual_view", "value"),
    Input("switches_mask_single", "value"),
)
def update_visual_control_panel(clickData, selected_models, view, model_values):
    if clickData is None or view is None:
        raise PreventUpdate
    else:
        current_model = clickData["points"][0]["x"]
        options_multi = []
        if view == "multiple":
            counter = 3
            for m in selected_models:
                if m != current_model:
                    options_multi.append({"label": m, "value": counter})
                    counter += 1
        # values = model_values
        return json.dumps({"slice_id": clickData["points"][0]["y"],
                           "model": current_model,
                           "metric_value": clickData["points"][0]["y"]}), options_multi


@app.callback(
    Output("df_metric_detail", "data"),
    Output("dropdown_detail_model", "value"),
    Input("df_total", "data"),
    Input("df_patient_id", "data"),
    Input("submit_detail", "n_clicks"),
    Input("dropdown_detail_metric", "value"),
    State("dropdown_detail_model", "value"),
    Input("radioitem_detail_slices", "value"))
def update_data_detail(info_total, patient_id, n_clicks, metric, models, slice_type):
    if info_total is None or patient_id is None:
        raise PreventUpdate
    elif models is None or len(models) == 0:
        return pd.DataFrame().to_json(), []
    else:
        patient_id = int(json.loads(patient_id))  # int(clickData["points"][0]["y"])
        df = pd.read_json(f"/tf/workdir/data/VS_segm/VS_registered/test_processed/vs_gk_{patient_id}/evaluation.json")

        if metric in ["DSC", "ASSD"]:
            lookup = {"DSC": "dice", "ASSD": "assd"}
            cols = [c for c in df.columns if lookup[metric] in c]
            df_metric = df[["slice", "VS_class_gt"] + cols]
            df_metric.rename(columns={k: k.split("-")[-1] for k in cols}, inplace=True)
            models_selected = get_selected_model_list(models, fixed=["slice", "VS_class_gt"])
            if models_selected is not None:
                df_metric = df_metric[models_selected]
            if slice_type == "only_tumor":
                df_metric = df_metric[df_metric["VS_class_gt"] == 1]
            df_metric.drop(columns=["VS_class_gt"], inplace=True)
            df_metric = df_metric.append({"slice": metric, **dict(df_metric.mean()[1:])}, ignore_index=True)

        else:  # ["ACC", "TPR", "TNR"]
            cols = [c for c in df.columns if "class_pred-" in c]
            df_metric = df[["slice", "VS_class_gt"] + cols]
            df_metric.rename(columns={k: k.split("-")[-1] for k in cols}, inplace=True)
            models_selected = get_selected_model_list(models, fixed=["slice", "VS_class_gt"])
            if models_selected is not None:
                df_metric = df_metric[models_selected]
            if slice_type == "only_tumor":
                df_metric = df_metric[df_metric["VS_class_gt"] == 1]
            lookup = {"ACC": TestSet().calculate_accuracy,
                      "TPR": TestSet().calculate_tpr,
                      "TNR": TestSet().calculate_tnr}
            model_cols = list(df_metric.columns)[2:]
            values = [lookup[metric](
                confusion_matrix(df_metric["VS_class_gt"].values, x[1].values, labels=[0, 1]).ravel()) for x in
                df_metric[model_cols].items()]
            df_metric.drop(columns=["VS_class_gt"], inplace=True)
            df_metric = df_metric.append(
                {"slice": metric, **{k: v for k, v in zip(model_cols, values)}}, ignore_index=True)

        models_selected = list(df_metric.columns)[1:]
        return df_metric.to_json(), models_selected


@app.callback(
    Output("detail_plot", "figure"),
    Input("df_metric_detail", "data"),
    Input("detail_plot_slider", "value"),
    Input("detail_plot_slider", "max"), )
def update_heatmap_detail(info, slider_values, slider_max):
    if info is None:
        raise PreventUpdate
    else:
        # read dataframe
        df_metric = pd.read_json(info)
        if len(df_metric) == 0:
            return fig_no_model_selected
        metric = df_metric.iloc[-1]["slice"]

        # define colorscale and tickvals
        colorscale, tickvals = get_colorscale_tickvals(metric, slider_values, slider_max)

        # create figure
        fig = make_subplots(rows=2, cols=1,
                            row_heights=[0.1, 0.9], vertical_spacing=0.05, shared_xaxes=True)
        # create annotated heatmap with total values
        round_dec = 2 if len(df_metric.columns) >= 8 else 3
        trace = ff.create_annotated_heatmap(x=list(df_metric.columns)[1:],
                                            y=["mean"],
                                            z=[list(df_metric.iloc[-1][1:].values)],
                                            hoverinfo='skip',
                                            coloraxis="coloraxis",
                                            annotation_text=[
                                                [np.round(x, round_dec) for x in list(df_metric.iloc[-1][1:].values)]])
        fig.add_trace(trace.data[0],
                      1, 1)
        fig.layout.update(trace.layout)

        # prepare x,y,z for heatmap
        x = list(df_metric.columns)[1:]
        y = list(df_metric["slice"].values[:-1])
        z = [list(df_metric.iloc[idx][1:].values) for idx in range(len(df_metric) - 1)]
        # create hovertext
        hovertext = list()
        for yi, yy in enumerate(y):
            hovertext.append(list())
            for xi, xx in enumerate(x):
                hovertext[-1].append(
                    'Model: {}<br />Slice: {}<br />{}: {}'.format(xx, yy, metric, np.round(z[yi][xi], decimals=5)))
        # heatmap for patient data
        fig.add_trace(go.Heatmap(x=x,
                                 y=y,
                                 z=z,
                                 hoverongaps=True,
                                 hoverinfo='text',
                                 text=hovertext,
                                 coloraxis="coloraxis"), 2, 1)
        # update layout
        fig.update_layout(xaxis2={'showticklabels': False},
                          xaxis1={'side': 'top', 'showticklabels': True},
                          yaxis2={'title': 'Slice'},
                          coloraxis={'colorscale': colorscale,
                                     'colorbar': dict(title=metric, tickvals=tickvals, tickmode="array")},
                          margin=dict(l=5,
                                      r=5,
                                      b=5,
                                      t=5,
                                      pad=4)
                          )

        return fig


@app.callback(
    Output("df_2d_data", "data"),
    Output("visual_model_slider", "marks"),
    Output("switches_mask_single", "options"),
    Input("df_patient_id", "data"),
    Input("visual_slice_slider", "value"),
    Input("visual_model_slider", "value"),
    Input("dropdown_detail_model", "value"),
)
def update_data_2d(info_patient_id, slice_value, model_value, model_list):
    if info_patient_id is None:
        raise PreventUpdate
    else:
        patient_id = int(json.loads(info_patient_id))
        slice = slice_value
        model = model_list[model_value]
        print("model at data update time", model_value, model)
        marks = {idx: {'label': ''} if idx != model_value else {'label': m} for idx, m in enumerate(model_list)}
        options_single = [{"label": "GT", "value": 1}, {"label": model, "value": 2}]
        df = pd.read_json(f"/tf/workdir/data/VS_segm/VS_registered/test_processed/vs_gk_{patient_id}/evaluation.json")
        container = DataContainer(f"/tf/workdir/data/VS_segm/VS_registered/test_processed/vs_gk_{patient_id}/")
        img = container.t2_scan_slice(slice)
        if df.iloc[slice][f"VS_class_pred-{model}"] == 1:
            segm = cv2.drawContours(np.zeros((256, 256)),
                                    [np.array(s) for s in df.iloc[slice][f"VS_segm_pred-{model}"]], -1, (1, 1, 1), 1)
        else:
            segm = np.zeros((256, 256))
        if df.iloc[slice]["VS_class_gt"] == 1:
            gt = cv2.drawContours(np.zeros((256, 256)), [np.array(s) for s in df.iloc[slice]["VS_segm_gt"]], -1,
                                  (1, 1, 1), -1)
        else:
            gt = np.zeros((256, 256))
        df_2d = pd.DataFrame()
        df_2d = df_2d.append({"slice": slice, "img": img, "segm_gt": gt, "class_gt": df.iloc[slice]["VS_class_gt"],
                              "segm_pred": segm, "class_pred": df.iloc[slice][f"VS_class_pred-{model}"]},
                             ignore_index=True)
        return df_2d.to_json(), marks, options_single


@app.callback(
    Output("visual_plot", "figure"),
    Input("df_2d_data", "data"),
    Input("switches_mask_single", "value")
)
def update_visual_plot_2d(info, selected_masks):
    if info is None:
        raise PreventUpdate
    else:
        df = pd.read_json(info)
        colorscale_segm = [[0, 'rgba(0,0,0,0)'],
                           [0.99, 'rgba(0,0,0,0)'],
                           [1.0, 'rgba(255,0,0,1)']]  # 165,0,38,1 - RdYlBu[0]
        colorscale_gt = [[0, 'rgba(0,0,0,0)'],
                         [0.99, 'rgba(0,0,0,0)'],
                         [1, 'rgba(0,0,255,1)']]  # 49,54,149,1 - RdYlBu[-1]
        fig = go.Figure()
        fig_img = px.imshow(np.array(df.iloc[0]["img"]), binary_string=True)
        fig.add_trace(fig_img.data[0])
        fig.update_traces(opacity=1.0)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        print("selected mask", selected_masks)
        if 1 in selected_masks:
            fig.add_heatmap(z=np.array(df.iloc[0]["segm_gt"]), hoverinfo="skip", showscale=False,
                            colorscale=colorscale_gt, opacity=0.4)
        if 2 in selected_masks:
            fig.add_heatmap(z=np.array(df.iloc[0]["segm_pred"]), hoverinfo="skip", showscale=False,
                            colorscale=colorscale_segm, opacity=0.8)
        fig.update_layout(margin=dict(l=5,
                                      r=5,
                                      b=5,
                                      t=5,
                                      pad=4)
                          )
        return fig


# @app.callback(
#     Output("visual_plot", "figure"),
#     Input("df_total", "data"),
#     Input("data_detail", "data"),
#     Input("detail_plot", "clickData"))
# def update_visual_plot(info_total, info_detail, clickData):
#     if info_total is None or clickData is None:
#         raise PreventUpdate
#     else:
#         slice_id = int(clickData["points"][0]["y"])
#         model_type = clickData["points"][0]["x"]
#         patient_id = int(json.loads(info_detail)["patient_id"])
#         df_total = pd.read_json(info_total)
#         df_slice = pd.DataFrame(df_total.iloc[patient_id - 200]["values"])
#         container = DataContainer(f"/tf/workdir/data/VS_segm/VS_registered/test/vs_gk_{patient_id}/")
#         segm_contour = df_slice[f"VS_segm_pred-{model_type}"].iloc[slice_id]
#         segm = cv2.drawContours(np.zeros((256, 256)), [np.array(s) for s in segm_contour], -1, (1, 1, 1), -1).astype(
#             np.float32)
#         img = container.t2_scan_slice(slice_id)
#         toshow = img
#         toshow[segm == 1] = 1.1
#         fig = px.imshow(toshow, color_continuous_scale=[*px.colors.sequential.gray, '#fb9f3a'])
#         fig.update_layout(title=f"Patient ID: {patient_id}, Slice: {slice_id}, Model:  {model_type}")
#         fig.update_xaxes(showticklabels=False)
#         fig.update_yaxes(showticklabels=False)
#         return fig


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8060, debug=True)
