import io, base64
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import numpy as np
from PIL import Image
import plotly.express as px
import tensorflow as tf
import shap
from lime import lime_image
import cv2

# Your implementations
from steps.lime_explanation import LIMEExplainer
from steps.gradcam_explanation import GradCAMExplainer
from steps.shap_explanation import SHAPExplainer
from steps.inference import inference

# Load model once
model = tf.keras.models.load_model("../models/my_model_2.keras")
class_names = ["NORMAL", "PNEUMONIA"]

# Initialize explainers
lime_explainer = LIMEExplainer(model)
gradcam_explainer = GradCAMExplainer(model)
shap_explainer = SHAPExplainer(model)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "XAI Comparison Dashboard"

UPLOAD_STYLE = {
    'width': '100%',
    'height': '150px',
    'lineHeight': '150px',
    'borderWidth': '2px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px 0',
    'cursor': 'pointer',
    'background': '#f8f9fa'
}

CARD_STYLE = {
    'borderRadius': '10px',
    'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
    'padding': '15px',
    'margin': '10px 0',
    'background': 'white'
}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Explainable AI Dashboard", className="text-center my-4"),
            html.P("Compare different XAI methods for medical image analysis", 
                  className="text-center text-muted mb-4"),
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Upload Image", className="h5"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt fa-3x mb-2"),
                            html.P('Drag and Drop or Click to Select', className="lead")
                        ]),
                        style=UPLOAD_STYLE,
                        multiple=False
                    ),
                    dbc.Button(
                        "Generate Explanations", 
                        id="btn-explain", 
                        color="primary", 
                        className="mt-3 w-100",
                        disabled=True
                    ),
                    dbc.Alert(
                        "Please upload an image first",
                        id="upload-alert",
                        color="warning",
                        dismissable=True,
                        is_open=False,
                        className="mt-3"
                    )
                ])
            ], style=CARD_STYLE),

            dbc.Card([
                dbc.CardHeader("Prediction Results", className="h5"),
                dbc.CardBody([
                    html.Div(id="pred-info", className="h4 text-center"),
                    dbc.Progress(id="pred-confidence", className="my-3"),
                    html.Div([
                        dbc.Badge("LIME", color="info", className="me-1"),
                        dbc.Badge("SHAP", color="success", className="me-1"),
                        dbc.Badge("Grad-CAM", color="danger")
                    ], className="text-center mt-2")
                ])
            ], style=CARD_STYLE)
        ], width=4),

        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Original Image", tab_id="original"),
                dbc.Tab(label="LIME Explanation", tab_id="lime"),
                dbc.Tab(label="SHAP Explanation", tab_id="shap"),
                dbc.Tab(label="Grad-CAM Explanation", tab_id="gradcam"),
            ], id="tabs", active_tab="original"),

            dbc.Card([
                dbc.CardBody([
                    html.Div(id="tab-content", className="text-center")
                ])
            ], style={**CARD_STYLE, 'minHeight': '500px'})
        ], width=8)
    ]),

    # Hidden stores for explanations
    dcc.Store(id="store-lime"),
    dcc.Store(id="store-shap"),
    dcc.Store(id="store-gradcam"),

    dcc.Loading(
        id="loading",
        type="circle",
        fullscreen=True,
        children=html.Div(id="loading-output")
    )
], fluid=True)

@app.callback(
    Output("btn-explain", "disabled"),
    Output("upload-alert", "is_open"),
    Input("upload", "contents")
)
def toggle_explain_button(contents):
    if contents is None:
        return True, True
    return False, False

@app.callback(
    Output("pred-info", "children"),
    Output("pred-confidence", "value"),
    Output("pred-confidence", "label"),
    Output("pred-confidence", "color"),
    Output("store-lime", "data"),
    Output("store-shap", "data"),
    Output("store-gradcam", "data"),
    Input("btn-explain", "n_clicks"),
    State("upload", "contents"),
    prevent_initial_call=True
)
def explain(n_clicks, contents):
    if not contents or n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, None, None, None

    prefix, content = contents.split(",")
    img = Image.open(io.BytesIO(base64.b64decode(content))).resize((256,256)).convert("RGB")
    arr = np.array(img).astype(np.float32)

    img_batch, preds = inference(model, arr, normalize=True)
    score = float(preds[0][0] if preds.shape[-1]==1 else max(preds[0]))
    label = class_names[int(score>0.5)] if preds.shape[-1]==1 else class_names[np.argmax(preds[0])]

    confidence = abs(score - 0.5) * 2
    progress_color = "danger" if confidence < 0.7 else "warning" if confidence < 0.9 else "success"

    lime_res = lime_explainer.explain(arr, num_samples=100)
    shap_explanation = shap_explainer.explain(arr)
    shap_values = shap_explanation["shap_values"]
    shap_positive = shap_values[:, :, :, 1]
    shap_sum = np.sum(shap_positive, axis=2)
    gradcam_res = gradcam_explainer.explain(arr)

    return (
        f"Diagnosis: {label}",
        confidence * 100,
        f"{score:.2f} confidence",
        progress_color,
        lime_res,
        shap_sum.tolist(),
        gradcam_res
    )

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    State("upload", "contents"),
    State("store-lime", "data"),
    State("store-shap", "data"),
    State("store-gradcam", "data"),
)
def update_tab(active_tab, contents, lime_res, shap_sum, gradcam_res):
    if not contents:
        return dash.no_update

    prefix, content = contents.split(",")
    img = Image.open(io.BytesIO(base64.b64decode(content))).resize((256,256)).convert("RGB")
    arr = np.array(img).astype(np.float32)

    return get_tab_content(active_tab, arr, lime_res, shap_sum, gradcam_res)

def get_tab_content(tab_id, original_img, lime_res, shap_heatmap, gradcam_res):
    if tab_id == "original":
        fig = px.imshow(original_img)
        title = "Original Image"
    elif tab_id == "lime":
        fig = px.imshow(lime_res["superimposed_image"])
        title = "LIME Explanation"
    elif tab_id == "shap":
        fig = px.imshow(np.array(shap_heatmap), color_continuous_scale='hot')
        title = "SHAP Heatmap"
    elif tab_id == "gradcam":
        fig = px.imshow(gradcam_res["superimposed_image"])
        title = "Grad-CAM Explanation"

    fig.update_layout(
        title={'text': title, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return dcc.Graph(
        figure=fig,
        config={'displayModeBar': False},
        style={'height': '500px'}
    )

if __name__ == "__main__":
    app.run(debug=True)
