########################################################################################################################
# Visualization application with DASH+Plotly
########################################################################################################################

import ast
import json
import cv2
import pandas as pd
import numpy as np
import dash
from _plotly_utils.colors import n_colors
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
# app.config['suppress_callback_exceptions'] = True

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
load_data = html.Div(
    id="load_data",
    children=[html.Div(children=[dbc.Button('Load Data', color='primary', size='sm', id='button-load-data')],
                       style={'width': '30%', 'display': 'inline-block'}),
              html.Div(children=[dcc.Loading(id='loading-data', type="default",
                                             children=html.Div(id='loading-data-output',
                                                               style={'color': 'lightgray'}))],
                       style={'width': '70%', 'display': 'inline-block', 'textAlign': 'left'}),
              dcc.Store(id='df-total'),
              dcc.Store(id='df-signature')
              ],
    style={'width': '15%',
           'verticalAlign': 'center',
           'display': 'inline-block'}
)
header1 = html.Div(
    id='header-1',
    children=html.H1(children='Domain Adaptation Segmentation Evaluation',
                     style={'textAlign': 'center', 'width': '66%', 'color': 'white'}),
    style={'width': '75%', 'display': 'inline-block'}
)
github_link = html.Div(
    id='github_link',
    children=[html.Img(id="logo", src=app.get_asset_url("dash-logo.png"),
                       width='50%', style={'verticalAlign': 'center'}),
              dbc.Button("Github Repo",
                         id='github_link_redirect',
                         href='https://github.com/CarolineMagg/DA_brain',
                         style={'textAlign': 'center'}
                         )],
    style={'width': '10%', 'display': 'inline-block'}
)
div_header = html.Div(
    id='header',
    children=[load_data,
              header1,
              github_link],
    style={'width': '100%',
           'backgroundColor': 'black'}
)

# control panel
control_error_metric_dataset = html.Div(
    children=[html.Div(children=html.Label("Error Metric:"),
                       style={'width': '10%',
                              'textAlign': 'left',
                              'paddingLeft': '1%',
                              'display': 'table-cell'}),
              html.Div(children=dcc.Dropdown(id='dropdown-error-metric',
                                             options=[{'label': 'DSC (DiceSimilarityCoefficient)',
                                                       'value': 'DSC'},
                                                      {'label': 'ASSD (AverageSymmetricSurfaceDistance',
                                                       'value': 'ASSD'},
                                                      {'label': 'ACC (Accuracy)',
                                                       'value': 'ACC'},
                                                      {'label': 'TPR (TruePositiveRate)',
                                                       'value': 'TPR'},
                                                      {'label': 'TNR (TrueNegativeRate)',
                                                       'value': 'TNR'}],
                                             value='DSC'
                                             ),
                       style={'width': '40%',
                              'display': 'table-cell'}),
              html.Div(children=html.Label("Dataset:"),
                       style={'width': '10%',
                              'textAlign': 'right',
                              'paddingRight': '1%',
                              'display': 'table-cell'}),
              html.Div(children=[dcc.RadioItems(id="radioitem-dataset",
                                                options=[{'label': 'only tumor',
                                                          'value': 'only_tumor'},
                                                         {'label': 'all',
                                                          'value': 'all'}],
                                                value='only_tumor')],
                       style={'width': '40%',
                              'textAlign': 'left',
                              'display': 'table-cell'})

              ],
    style={'display': 'table',
           'width': '97%'})
control_model = html.Div(
    children=[html.Div(children=html.Label("Model:"),
                       style={'width': '10%',
                              'textAlign': 'left',
                              'verticalAlign': 'middle',
                              'paddingLeft': '1%',
                              'display': 'table-cell'}),
              html.Div(children=dcc.Dropdown(id="dropdown-model",
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
                       style={'width': '80%',
                              'display': 'table-cell',
                              'font-size': '85%'}),
              html.Div(children=html.Button(id='submit-model', n_clicks=0, children="Apply"),
                       style={'width': '10%',
                              'display': 'table-cell',
                              'verticalAlign': 'center',
                              'align': 'center'})
              ],
    style={'display': 'table',
           'width': '97%'})
div_control_panel_heatmap = html.Div(
    children=[control_error_metric_dataset,
              html.Br(),
              control_model,
              dcc.Store(id='df-metric-overview'),
              dcc.Store(id='df-metric-detail'),
              dcc.Store(id='dict-patient-id'),
              dcc.Store(id='dict-slice-id'),
              dcc.Store(id='json-selected-models')],
    style={'display': 'table-cell',
           'width': '66%'})

control_prediction = html.Div(
    children=[html.Div(children=html.Label("Mask:"),
                       style={'width': '10%',
                              'textAlign': 'left',
                              'paddingLeft': '1%',
                              'display': 'table-cell'}),
              html.Div(children=dcc.RadioItems(id='radioitem-slice-view',
                                               options=[{'label': 'sum',
                                                         'value': 'sum'},
                                                        {'label': 'subtraction',
                                                         'value': 'subtraction'}],
                                               value='sum'),
                       style={'width': '30%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=html.Label("Opacity:"),
                       style={'width': '15%',
                              'textAlign': 'left',
                              'paddingLeft': '1%',
                              'display': 'table-cell'}),
              html.Div(children=dcc.Slider(id='slider-mask-opacity',
                                           min=0,
                                           max=1,
                                           step=0.1,
                                           value=0.8,
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           ),
                       style={'width': '45%',
                              'verticalAlign': 'middle',
                              'display': 'table-cell'})],
    style={'display': 'table',
           'width': '97%'})
control_gt = html.Div(
    children=[html.Div(children=html.Label("GT:"),
                       style={'width': '10%',
                              'textAlign': 'left',
                              'paddingLeft': '1%',
                              'display': 'table-cell'}),
              html.Div(children=[dcc.RadioItems(id='radioitem-gt-toggle',
                                                options=[{'label': 'show',
                                                          'value': 'show'},
                                                         {'label': 'hide',
                                                          'value': 'hide'}],
                                                value='hide')],
                       style={'width': '90%',
                              'textAlign': 'left',
                              'display': 'table-cell'})],
    style={'display': 'table',
           'width': '97%'})
control_gt_2 = html.Div(
    id="control_gt_2",
    children=[html.Div(children=[],
                       style={'width': '10%',
                              'paddingLeft': '1%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=dcc.RadioItems(id='radioitem-gt-type',
                                               options=[{'label': 'contour',
                                                         'value': 'contour'},
                                                        {'label': 'mask',
                                                         'value': 'mask'}],
                                               value="contour"),
                       style={'width': '30%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=html.Label("Opacity:"),
                       style={'width': '15%',
                              'textAlign': 'left',
                              'paddingLeft': '1%',
                              'display': 'table-cell'}),
              html.Div(children=dcc.Slider(id='slider-gt-opacity',
                                           min=0,
                                           max=1,
                                           step=0.1,
                                           value=0.9,
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           ),
                       style={'width': '45%',
                              'verticalAlign': 'middle',
                              'display': 'table-cell'})],
    style={'display': 'none',
           'width': '97%'})

div_slice_control_panel = html.Div(
    children=[control_prediction,
              html.Br(),
              control_gt,
              html.Br(),
              control_gt_2],
    style={'display': 'table-cell',
           'width': '33%'}
)
control_panel = html.Div(
    id="control_panel",
    style={'width': '100%',
           'backgroundColor': 'darkgray',
           'display': 'table'},
    children=[div_control_panel_heatmap,
              div_slice_control_panel,
              dcc.Store(id='dict-slice-data')])

sub_headers = html.Div(
    id='sub-headers',
    children=[html.Div(children=html.H2("All Patients", id='header-overview', style={'textAlign': 'center'}),
                       style={'width': '33.3%', 'display': 'table-cell'}),
              html.Div(children=html.H2("Patient", id='header-detail', style={'textAlign': 'center'}),
                       style={'width': '33.3%', 'display': 'table-cell'}),
              html.Div(children=html.H2("Slice", id='header-slice', style={'textAlign': 'center'}),
                       style={'width': '33.3%', 'display': 'table-cell'}),
              ],
    style={'width': '100%',
           'backgroundColor': 'lightgray',
           'borderColor': 'black',
           'display': 'table'},
)

# heatmaps
heatmap_1 = html.Div(
    children=[dcc.Graph(id='heatmap-overview',
                        figure=fig_no_data_available),
              dcc.RangeSlider(id='heatmap-overview-slider',
                              min=0,
                              max=1,
                              step=0.01,
                              value=[0, 1],
                              tooltip={'placement': 'bottom',
                                       'always_visible': True},
                              allowCross=False)],
    style={'display': 'table-cell',
           'width': '33.3%'}
)
heatmap_2 = html.Div(
    children=[dcc.Graph(id='heatmap-detail', figure=fig_no_data_selected),
              dcc.RangeSlider(id='heatmap-detail-slider',
                              min=0,
                              max=1,
                              step=0.01,
                              value=[0, 1],
                              tooltip={'placement': 'bottom',
                                       'always_visible': True},
                              allowCross=False)],
    style={'display': 'table-cell',
           'width': '33.3%'}
)
slice_plot = html.Div(
    children=[
        # html.Div(children=[dcc.Slider(id="visual_slice_slider",
        #                                     min=0,
        #                                     max=80,
        #                                     step=1,
        #                                     value=0,
        #                                     tooltip={'always_visible': True},
        #                                     vertical=True)],
        #                style={'width': "8%", 'height': "96%",
        #                       'display': "inline-block", 'position': "relative", "margin-bottom": "5%"}),
        html.Div(children=[  # dcc.Slider(id='visual_model_slider',
            #           min=0,
            #           max=len(MODELS) + 1,
            #           step=None,
            #           value=0,
            #           vertical=False),
            dcc.Graph(id='slice-plot', figure=fig_no_slice_selected)],
            style={'width': '100%', 'height': '100%',
                   'display': "inline-block", 'position': "relative"})],
    style={'display': 'table-cell',
           'width': '33.3%'}
)
first_row = html.Div(
    id='heatmaps',
    style={'width': '98%',
           'display': 'table'},
    children=[heatmap_1,
              heatmap_2,
              slice_plot]
)

# parallel set plots
div_overview_pc = html.Div(
    children=[dcc.Graph(id="overview_pc",
                        figure=fig_no_data_available)],
    style={'display': 'table-cell',
           'width': '50%'}
)
div_detail_pc = html.Div(
    children=[dcc.Graph(id="detail_pc", figure=fig_no_data_selected)],
    style={'display': 'table-cell',
           'width': '50%'}
)

second_row = html.Div(
    id='parallel-set-plots',
    style={'width': '100%',
           'display': 'table',
           'backgroundColor': 'grey'},
    children=[div_overview_pc,
              div_detail_pc]
)

# layout
app.layout = html.Div(
    children=[div_header,
              sub_headers,
              control_panel,
              first_row,
              second_row
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


def get_selected_model_list(models):
    models_selected = []
    if "All" not in models:
        models_selected = models
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
    if "All" in models:
        models_selected = None
    return models_selected


@app.callback(
    Output('loading-data-output', "children"),
    Output('df-total', "data"),
    Output('df-signature', "data"),
    Input('load_data', "n_clicks"))
def load_data_spinner(n_clicks):
    # if n_clicks == 0 or n_clicks is None:
    #     raise PreventUpdate
    # else:
    testset = TestSet("/tf/workdir/data/VS_segm/VS_registered/test_processed/", load=True,
                      data_load=False, evaluation_load=False, radiomics_load=False)
    df_total = testset.df_total
    df_signature = testset.df_signature
    return "Data is loaded.", df_total.to_json(), df_signature.to_json()


@app.callback(
    Output('heatmap-overview-slider', "step"),
    Output('heatmap-overview-slider', "max"),
    Output('heatmap-overview-slider', "value"),
    Input('dropdown-error-metric', "value"))
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
    Output('heatmap-detail-slider', "step"),
    Output('heatmap-detail-slider', "max"),
    Output('heatmap-detail-slider', "value"),
    Input('dropdown-error-metric', "value"))
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
    Output('json-selected-models', "data"),
    Output('dropdown-model', "value"),
    Input('submit-model', "n_clicks"),
    State('dropdown-model', "value"),
)
def update_selected_models(n_clicks, selected_models):
    if selected_models is None or len(selected_models) == 0:
        return json.dumps([]), []
    else:
        selected_models_processed = get_selected_model_list(selected_models)
        if selected_models_processed is None:
            selected_models_processed = MODELS
        return json.dumps(selected_models_processed), selected_models_processed


@app.callback(
    Output('df-metric-overview', "data"),
    Input('df-total', "data"),
    Input('json-selected-models', "data"),
    Input('dropdown-error-metric', "value"),
    Input('radioitem-dataset', "value"))
def update_data_overview(json_df_total, json_selected_models, selected_metric, selected_dataset):
    if json_df_total is None or json_selected_models is None:
        raise PreventUpdate
    else:
        df_total = pd.read_json(json_df_total)
        selected_models = json.loads(json_selected_models)
        if selected_models is not None and len(selected_models) == 0:
            return pd.DataFrame().to_json()
        lookup = {"DSC": f"dice_{selected_dataset}",
                  "ASSD": f"assd_{selected_dataset}",
                  "ACC": f"acc_{selected_dataset}",
                  "TPR": f"tpr_{selected_dataset}",
                  "TNR": f"tnr_{selected_dataset}"}
        df_metric = pd.DataFrame(df_total.iloc[0][lookup[selected_metric]])
        models_selected = ["id"] + selected_models if selected_models is not None else None
        if models_selected is not None:
            df_metric = df_metric[models_selected]
        return df_metric.to_json()


@app.callback(
    Output('heatmap-overview', "figure"),
    Input('df-metric-overview', "data"),
    Input('heatmap-overview-slider', "value"),
    Input('heatmap-overview-slider', "max"))
def update_heatmap_overview(json_df_metric, slider_values, slider_max):
    if json_df_metric is None:
        raise PreventUpdate
    else:
        # read dataframe
        df_metric = pd.read_json(json_df_metric)
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
                                                [np.round(x, round_dec) for x in
                                                 list(df_metric.iloc[-1][1:].values)]])
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
    Output('dict-patient-id', "data"),
    Output('header-detail', "children"),
    Input('heatmap-overview', "clickData"))
def update_patient_id_detail(clickData):
    if clickData is None:
        raise PreventUpdate
    else:
        patient_id = clickData["points"][0]["y"]
        return json.dumps(patient_id), f"Patient ID {patient_id}"


@app.callback(
    Output('dict-slice-id', "data"),
    Output('header-slice', "children"),
    Input('heatmap-detail', "clickData")
)
def update_slice_id_detail(clickData):
    if clickData is None:
        raise PreventUpdate
    else:
        slice_id = clickData["points"][0]["y"]
        return json.dumps(slice_id), f"Slice ID {slice_id}"


@app.callback(
    Output('df-metric-detail', "data"),
    Input('dict-patient-id', "data"),
    Input('json-selected-models', "data"),
    Input('dropdown-error-metric', "value"),
    Input('radioitem-dataset', "value"))
def update_data_detail(json_patient_id, json_selected_models, selected_metric, selected_dataset):
    if json_patient_id is None:
        raise PreventUpdate
    else:
        patient_id = int(json.loads(json_patient_id))  # int(clickData["points"][0]["y"])
        df = pd.read_json(f"/tf/workdir/data/VS_segm/VS_registered/test_processed/vs_gk_{patient_id}/evaluation.json")
        selected_models = json.loads(json_selected_models)
        if selected_models is not None and len(selected_models) == 0:
            return pd.DataFrame().to_json()
        if selected_metric in ["DSC", "ASSD"]:
            lookup = {"DSC": "dice", "ASSD": "assd"}
            cols = [c for c in df.columns if lookup[selected_metric] in c]
            df_metric = df[["slice", "VS_class_gt"] + cols]
            df_metric.rename(columns={k: k.split("-")[-1] for k in cols}, inplace=True)
            models_selected = ["slice", "VS_class_gt"] + selected_models if selected_models is not None else None
            if models_selected is not None:
                df_metric = df_metric[models_selected]
            if selected_dataset == "only_tumor":
                df_metric = df_metric[df_metric["VS_class_gt"] == 1]
            df_metric.drop(columns=["VS_class_gt"], inplace=True)
            df_metric = df_metric.append({"slice": selected_metric, **dict(df_metric.mean()[1:])}, ignore_index=True)

        else:  # ["ACC", "TPR", "TNR"]
            cols = [c for c in df.columns if "class_pred-" in c]
            df_metric = df[["slice", "VS_class_gt"] + cols]
            df_metric.rename(columns={k: k.split("-")[-1] for k in cols}, inplace=True)
            models_selected = ["slice", "VS_class_gt"] + selected_models if selected_models is not None else None
            if models_selected is not None:
                df_metric = df_metric[models_selected]
            if selected_dataset == "only_tumor":
                df_metric = df_metric[df_metric["VS_class_gt"] == 1]
            lookup = {"ACC": TestSet().calculate_accuracy,
                      "TPR": TestSet().calculate_tpr,
                      "TNR": TestSet().calculate_tnr}
            model_cols = list(df_metric.columns)[2:]
            values = [lookup[selected_metric](
                confusion_matrix(df_metric["VS_class_gt"].values, x[1].values, labels=[0, 1]).ravel()) for x in
                df_metric[model_cols].items()]
            df_metric.drop(columns=["VS_class_gt"], inplace=True)
            df_metric = df_metric.append(
                {"slice": selected_metric, **{k: v for k, v in zip(model_cols, values)}}, ignore_index=True)

        return df_metric.to_json()


@app.callback(
    Output('heatmap-detail', "figure"),
    Input('df-metric-detail', "data"),
    Input('heatmap-detail-slider', "value"),
    Input('heatmap-detail-slider', "max"), )
def update_heatmap_detail(df_metric_json, slider_values, slider_max):
    if df_metric_json is None:
        raise PreventUpdate
    else:
        # read dataframe
        df_metric = pd.read_json(df_metric_json)
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


# @app.callback(
#     Output("df_2d_data", "data"),
#     Output("visual_model_slider", "marks"),
#     Output("switches_mask_single", "options"),
#     Input('dict-patient-id', "data"),
#     Input("visual_slice_slider", "value"),
#     Input("visual_model_slider", "value"),
#     Input("dropdown_detail_model", "value"),
# )
# def update_data_2d(info_patient_id, slice_value, model_value, model_list):
#     if info_patient_id is None:
#         raise PreventUpdate
#     else:
#         patient_id = int(json.loads(info_patient_id))
#         slice = slice_value
#         model = model_list[model_value]
#         print("model at data update time", model_value, model)
#         marks = {idx: {'label': ''} if idx != model_value else {'label': m} for idx, m in enumerate(model_list)}
#         options_single = [{"label": "GT", "value": 1}, {"label": model, "value": 2}]
#         df = pd.read_json(f"/tf/workdir/data/VS_segm/VS_registered/test_processed/vs_gk_{patient_id}/evaluation.json")
#         container = DataContainer(f"/tf/workdir/data/VS_segm/VS_registered/test_processed/vs_gk_{patient_id}/")
#         img = container.t2_scan_slice(slice)
#         if df.iloc[slice][f"VS_class_pred-{model}"] == 1:
#             segm = cv2.drawContours(np.zeros((256, 256)),
#                                     [np.array(s) for s in df.iloc[slice][f"VS_segm_pred-{model}"]], -1, (1, 1, 1), 1)
#         else:
#             segm = np.zeros((256, 256))
#         if df.iloc[slice]["VS_class_gt"] == 1:
#             gt = cv2.drawContours(np.zeros((256, 256)), [np.array(s) for s in df.iloc[slice]["VS_segm_gt"]], -1,
#                                   (1, 1, 1), -1)
#         else:
#             gt = np.zeros((256, 256))
#         df_2d = pd.DataFrame()
#         df_2d = df_2d.append({"slice": slice, "img": img, "segm_gt": gt, "class_gt": df.iloc[slice]["VS_class_gt"],
#                               "segm_pred": segm, "class_pred": df.iloc[slice][f"VS_class_pred-{model}"]},
#                              ignore_index=True)
#         return df_2d.to_json(), marks, options_single
#
#
# @app.callback(
#     Output('slice-plot', "figure"),
#     Input("df_2d_data", "data"),
#     Input("switches_mask_single", "value")
# )
# def update_visual_plot_2d(info, selected_masks):
#     if info is None:
#         raise PreventUpdate
#     else:
#         df = pd.read_json(info)
#         colorscale_segm = [[0, 'rgba(0,0,0,0)'],
#                            [0.99, 'rgba(0,0,0,0)'],
#                            [1.0, 'rgba(255,0,0,1)']]  # 165,0,38,1 - RdYlBu[0]
#         colorscale_gt = [[0, 'rgba(0,0,0,0)'],
#                          [0.99, 'rgba(0,0,0,0)'],
#                          [1, 'rgba(0,0,255,1)']]  # 49,54,149,1 - RdYlBu[-1]
#         fig = go.Figure()
#         fig_img = px.imshow(np.array(df.iloc[0]["img"]), binary_string=True)
#         fig.add_trace(fig_img.data[0])
#         fig.update_traces(opacity=1.0)
#         fig.update_xaxes(showticklabels=False)
#         fig.update_yaxes(showticklabels=False)
#         print("selected mask", selected_masks)
#         if 1 in selected_masks:
#             fig.add_heatmap(z=np.array(df.iloc[0]["segm_gt"]), hoverinfo="skip", showscale=False,
#                             colorscale=colorscale_gt, opacity=0.4)
#         if 2 in selected_masks:
#             fig.add_heatmap(z=np.array(df.iloc[0]["segm_pred"]), hoverinfo="skip", showscale=False,
#                             colorscale=colorscale_segm, opacity=0.8)
#         fig.update_layout(margin=dict(l=5,
#                                       r=5,
#                                       b=5,
#                                       t=5,
#                                       pad=4)
#                           )
#         return fig
#

@app.callback(
    Output('dict-slice-data', "data"),
    Input('dict-patient-id', "data"),
    Input('dict-slice-id', "data"),
    Input('json-selected-models', "data")
)
def update_data_slice(json_patient_id, json_slice_id, json_selected_models):
    if json_patient_id is None or json_slice_id is None or json_selected_models is None:
        raise PreventUpdate
    else:
        patient_id = int(json.loads(json_patient_id))
        slice = int(json.loads(json_slice_id))
        selected_models = json.loads(json_selected_models)
        if selected_models is not None and len(selected_models) == 0:
            return json.dumps({})
        if selected_models is None:
            selected_models = MODELS
        df = pd.read_json(f"/tf/workdir/data/VS_segm/VS_registered/test_processed/vs_gk_{patient_id}/evaluation.json")
        container = DataContainer(f"/tf/workdir/data/VS_segm/VS_registered/test_processed/vs_gk_{patient_id}/")
        img = container.t2_scan_slice(slice)
        segmentation_filled = []
        for m in selected_models:
            segm_filled = cv2.drawContours(np.zeros((256, 256)),
                                           [np.array(s).astype(np.int64) for s in
                                            np.array(df.iloc[slice][f"VS_segm_pred-{m}"], dtype="object")], -1, (1),
                                           -1)
            segmentation_filled.append(segm_filled)
        segm_sum = (np.sum(segmentation_filled, axis=0))
        z_max = len(segmentation_filled)
        gt_filled = cv2.drawContours(np.zeros((256, 256)),
                                     [np.array(s).astype(np.int64) for s in
                                      np.array(df.iloc[slice][f"VS_segm_gt"], dtype="object")], -1,
                                     (len(segmentation_filled)),
                                     -1)
        gt_contour = cv2.drawContours(np.zeros((256, 256)),
                                      [np.array(s).astype(np.int64) for s in
                                       np.array(df.iloc[slice][f"VS_segm_gt"], dtype="object")], -1,
                                      (len(segmentation_filled)), 1)
        segm_subtract = gt_filled - segm_sum
        info_dict = {"slice": slice,
                     "patient_id": patient_id,
                     "selected_models": selected_models,
                     "z_max": z_max,
                     "img": img.tolist(),
                     "segm_sum": segm_sum.tolist(),
                     "segm_subtract": segm_subtract.tolist(),
                     "gt_contour": gt_contour.tolist(),
                     "gt_filled": gt_filled.tolist()}
        return json.dumps(info_dict)


@app.callback(
    Output('control_gt_2', "style"),
    Input('radioitem-gt-toggle', "value"),
)
def update_gt_control(gt_toggle):
    if gt_toggle == "show":
        return {'display': 'table', 'width': '97%'}
    else:
        return {'display': 'none', 'width': '97%'}


@app.callback(
    Output('slice-plot', "figure"),
    Input('dict-slice-data', "data"),
    Input('radioitem-slice-view', "value"),
    Input('radioitem-gt-toggle', "value"),
    Input('radioitem-gt-type', "value"),
    Input('slider-mask-opacity', "value"),
    Input('slider-gt-opacity', "value")
)
def update_heatmap_slice(json_dict_slice_data, view_type, gt_toggle, gt_type, mask_opacity, gt_opacity):
    if json_dict_slice_data is None:
        raise PreventUpdate
    else:
        info_dict = json.loads(json_dict_slice_data)
        if len(info_dict) == 0:
            return fig_no_model_selected
        # image
        img = np.array(info_dict["img"])
        # gt
        gt = np.zeros_like(img)
        if gt_toggle == "show":
            if gt_type == "contour":
                gt = np.array(info_dict["gt_contour"])
            else:
                gt = np.array(info_dict["gt_filled"])
        # selected models
        z_max = int(info_dict["z_max"])
        selected_models = info_dict["selected_models"]
        if view_type == "sum":
            # select segm
            segm = np.array(info_dict["segm_sum"])
            # discrete colorscale
            bvals = list(range(0, z_max + 1, 1))
            nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]
            if z_max > 1:
                colors = n_colors('rgb(253,219,199)', 'rgb(103,0,31)', z_max, colortype='rgb')
            else:
                colors = ['rgb(103,0,31)']
            colors = [c.replace("rgb", "rgba").replace(")", ",1)") if "rgba" not in c else c for c in colors]
            # dcolorscale = []
            # for k in range(len(colors)):
            #     dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
            # # colorbar ticks
            # bvals2 = np.array(bvals)
            # # tickvals = np.linspace(-z_max + 1, z_max - 1, len(colors)).tolist()
            # if len(selected_models) < 5:
            #     tickvals = np.linspace(-z_max + 0.5, z_max - 0.5,
            #                            len(colors)).tolist()  # [np.mean(bvals[k:k+2]) for k in range(len(bvals)-1)] #
            # elif len(selected_models) >= 5:
            #     tickvals = np.linspace(-z_max + 0.5, z_max - 0.5, len(colors)).tolist()  # 1.75
            # # tickvals = [np.mean(bvals[k:k+2]) for k in range(len(bvals)-1)]
            # ticktext = [f"{k}" for k in bvals2[1:]]

        else:
            # select segm
            segm = np.array(info_dict["segm_subtract"])
            # discrete colorscale
            bvals = list(range(-z_max - 1, z_max + 1, 1))
            nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]
            if z_max > 1:
                red = n_colors('rgb(253,219,199)', 'rgb(103,0,31)', z_max, colortype='rgb')
                blue = n_colors('rgb(5,48,97)', 'rgb(209,229,240)', z_max, colortype='rgb')
            else:
                red = ['rgb(103,0,31)']
                blue = ['rgb(5,48,97)']
            colors = blue + ['rgba(255,255,255,1)'] + red
            colors = [c.replace("rgb", "rgba").replace(")", ",1)") if "rgba" not in c else c for c in colors]
            # dcolorscale = []
            # for k in range(len(colors)):
            #     dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
            # # colorbar ticks
            # bvals2 = np.array(bvals)
            # # if len(selected_models) < 5:
            # #     tickvals = np.linspace(-z_max + 0.5, z_max - 0.5,
            # #                            len(colors)).tolist()  # [np.mean(bvals[k:k+2]) for k in range(len(bvals)-1)] #
            # # elif len(selected_models) >= 5:
            # tickvals = np.linspace(-z_max + 1.5, z_max - 1.5, len(colors)).tolist()
            # # tickvals = [np.mean(bvals[k:k + 2]) for k in range(len(bvals) - 1)]
            # ticktext = [f"{k}" for k in bvals2[1:]]

        # colorscale gt
        colorscale_gt = [[0, 'rgba(0,0,0,0)'],
                         [0.99, 'rgba(0,0,0,0)'],
                         [1, 'rgba(0,0,0,1)']]

        # discrete colorscale segm
        dcolorscale = []
        for k in range(len(colors)):
            dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
        # colorbar ticks
        bvals2 = np.array(bvals)
        # if len(selected_models) < 5:
        #     tickvals = np.linspace(-z_max + 0.5, z_max - 0.5,
        #                            len(colors)).tolist()  # [np.mean(bvals[k:k+2]) for k in range(len(bvals)-1)] #
        # elif len(selected_models) >= 5:
        tickvals = np.linspace(-z_max + 1.5, z_max - 1.5, len(colors)).tolist()
        # tickvals = [np.mean(bvals[k:k + 2]) for k in range(len(bvals) - 1)]
        ticktext = [f"{k}" for k in bvals2[1:]]

        # hovertext
        segm[segm == 0] = None
        hovertext = list()
        for yi, yy in enumerate(segm):
            hovertext.append(list())
            for xi, xx in enumerate(segm[0]):
                hovertext[-1].append('value: {}'.format(segm[yi][xi]))

        # figure
        # print(ticktext)
        # print(tickvals)
        fig = go.Figure()
        fig_img = px.imshow(img, binary_string=True)
        fig.add_trace(fig_img.data[0])  # ,1,1)
        fig.update_traces(hovertemplate=None, hoverinfo="skip")
        fig.update_traces(opacity=1.0)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        fig.add_heatmap(z=segm, showscale=True, colorscale=dcolorscale,
                        zmid=0, hoverongaps=False, text=hovertext, hoverinfo='text',
                        colorbar=dict(thickness=30, tickmode="array", tickvals=tickvals, ticktext=ticktext),
                        opacity=mask_opacity, name="segm")

        fig.add_heatmap(z=gt, hoverinfo="skip", showscale=False, colorscale=colorscale_gt,
                        opacity=gt_opacity, name="gt")

        fig.update_layout(margin=dict(l=5, r=5, b=15, t=5, pad=4), uirevision=True)
        return fig


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8060, debug=True)
