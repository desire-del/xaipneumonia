import io, base64
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import numpy as np
from PIL import Image
import plotly.express as px
import tensorflow as tf
import cv2
from plotly import graph_objects as go
# Your implementations
from steps.lime_explanation import LIMEExplainer
from steps.gradcam_explanation import GradCAMExplainer
from steps.shap_explanation import SHAPExplainer
from steps.inference import inference
from src.log import logger

model = tf.keras.models.load_model("../models/my_model_2.keras")
class_names = ["NORMAL", "PNEUMONIA"]

lime_explainer = LIMEExplainer(model)
gradcam_explainer = GradCAMExplainer(model)
shap_explainer = SHAPExplainer(model)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "XAI Healthcare Dashboard"

UPLOAD_STYLE = {
    'width': '100%', 'height': '150px', 'lineHeight': '150px',
    'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '5px',
    'textAlign': 'center', 'margin': '10px 0', 'cursor': 'pointer', 'background': '#f5f5f5'
}

# Custom background styles
BACKGROUND_STYLE = {
    'background': 'linear-gradient(135deg, #f5f7fa 0%, #e4f0f8 100%)',
    'minHeight': '100vh',
    'padding': '20px'
}

# Card styling with glassmorphism effect
CARD_STYLE = {
    'background': 'rgba(255, 255, 255, 0.85)',
    'backdropFilter': 'blur(10px)',
    'borderRadius': '15px',
    'boxShadow': '0 8px 32px 0 rgba(31, 38, 135, 0.1)',
    'border': '1px solid rgba(255, 255, 255, 0.18)',
    'padding': '25px',
    'margin': '15px 0',
    'transition': 'all 0.3s ease'
}
LOADING_CONTAINER_STYLE = {
    'height': '500px',
    'position': 'relative',
    'border': '1px dashed #ddd',
    'borderRadius': '5px',
    'display': 'flex',
    'justifyContent': 'center',
    'alignItems': 'center',
    'backgroundColor': '#f9f9f9'
}

LOADING_STYLE = {
    'height': '100%',
    'width': '100%',
    'display': 'flex',
    'justifyContent': 'center',
    'alignItems': 'center'
}

app.layout = dbc.Container(
    style=BACKGROUND_STYLE,
    children=[
    dbc.Row([dbc.Col(html.H2("🩺 Explainable AI for Chest X-rays", className="text-center text-primary my-4"))]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Upload Chest X-ray Image", className="bg-primary text-white text-center"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt fa-3x mb-2 text-muted"),
                            html.P('Drag and Drop or Click to Upload')
                        ]),
                        style=UPLOAD_STYLE
                    ),
                ])
            ], style=CARD_STYLE),

            dbc.Card([
                dbc.CardHeader("Prediction Results", className="bg-info text-white text-center"),
                dbc.CardBody([
                    html.Div(id="pred-info", className="h5 text-center my-2 text-dark"),
                    dbc.Progress(id="pred-confidence", className="my-3", striped=True, animated=True),
                    html.Div([
                        dbc.Button("🔍 Explain with LIME", id="btn-lime", color="secondary", className="w-100 my-1", disabled=True),
                        dbc.Button("🧠 Explain with SHAP", id="btn-shap", color="success", className="w-100 my-1", disabled=True),
                        dbc.Button("🔥 Explain with Grad-CAM", id="btn-gradcam", color="danger", className="w-100 my-1", disabled=True)
                    ]),
                ])
            ], style=CARD_STYLE)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Visual Output", className="bg-secondary text-white text-center"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("📷 Original Image", className="text-muted text-center my-2"),
                            html.Div(id="original-image")
                        ], width=6),
                        dbc.Col([
                            html.H6("Explanation Output", className="text-muted text-center my-2"),
                            html.Div(
                                dcc.Loading(
                                    id="loading-explainer",
                                    type="circle",
                                    children=[html.Div(id="tab-content")],
                                    style=LOADING_STYLE
                                ),
                                style=LOADING_CONTAINER_STYLE
                            )
                        ], width=6)
                    ])
                ])
            ], style={**CARD_STYLE, 'minHeight': '600px'})
        ], width=9)
    ]),

    dcc.Store(id="store-img"),
    dcc.Store(id="store-lime"),
    dcc.Store(id="store-shap"),
    dcc.Store(id="store-gradcam")
], fluid=True)


@app.callback(
    Output("pred-info", "children"),
    Output("pred-confidence", "value"),
    Output("pred-confidence", "label"),
    Output("pred-confidence", "color"),
    Output("store-img", "data"),
    Output("btn-lime", "disabled"),
    Output("btn-shap", "disabled"),
    Output("btn-gradcam", "disabled"),
    Output("tab-content", "children"),
    Output("original-image", "children"),
    Input("upload", "contents")
)
def handle_upload(contents):
    if not contents:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, None, True, True, True, dash.no_update, dash.no_update

    prefix, content = contents.split(",")
    img = Image.open(io.BytesIO(base64.b64decode(content))).resize((256,256)).convert("RGB")
    arr = np.array(img).astype(np.float32)
    img_batch, preds = inference(model, arr, normalize=True)

    if preds.shape[-1] == 1:
        p_pneumonia = float(preds[0][0])
        p_normal = 1.0 - p_pneumonia
        label = class_names[int(p_pneumonia > 0.5)]
    else:
        p_normal, p_pneumonia = float(preds[0][0]), float(preds[0][1])
        label = class_names[np.argmax(preds[0])]

    score = p_pneumonia if label == "PNEUMONIA" else p_normal
    confidence = np.clip(abs(score - 0.5) * 2, 0, 1)
    color = "danger" if confidence < 0.7 else "warning" if confidence < 0.9 else "success"

    diagnosis_text = (
        f"Diagnosis: {label}  →  \xa0"
        f"P(NORMAL): {p_normal:.2%} — P(PNEUMONIA): {p_pneumonia:.2%}"
    )

    original_fig = px.imshow(arr.astype(np.uint8))
    original_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
    original_fig.update_xaxes(showticklabels=False)
    original_fig.update_yaxes(showticklabels=False)

    return diagnosis_text, confidence * 100, f"Confidence: {score:.2f}", color, arr.tolist(), False, False, False, "", dcc.Graph(figure=original_fig, config={'displayModeBar': False}, style={'height': '500px'})

@app.callback(
    Output("tab-content", "children", allow_duplicate=True),
    Output("store-lime", "data"),
    Input("btn-lime", "n_clicks"),
    State("store-img", "data"),
    prevent_initial_call=True
)
def explain_lime(n, img):
    arr = np.array(img, dtype=np.float32)
    res = lime_explainer.explain(arr, num_samples=1500)
    
    # Get the LIME mask and normalize it
    lime_mask = res["lime_mask"]
    lime_mask_normalized = (lime_mask - lime_mask.min()) / (lime_mask.max() - lime_mask.min())
    
    # Create a heatmap from the mask
    heatmap = np.uint8(255 * lime_mask_normalized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert original image to RGB (if grayscale)
    if len(arr.shape) == 2 or arr.shape[2] == 1:
        original_img = cv2.cvtColor(np.uint8(arr), cv2.COLOR_GRAY2RGB)
    else:
        original_img = np.uint8(arr)
    
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Blend the heatmap with original image
    superimposed_img = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)
    
    # Create the figure
    fig = px.imshow(superimposed_img)
    fig.update_layout(
        title="LIME Explanation",
        title_x=0.5,
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '500px'}), res

@app.callback(
    Output("tab-content", "children", allow_duplicate=True),
    Output("store-shap", "data", allow_duplicate=True),
    Input("btn-shap", "n_clicks"),
    State("store-img", "data"),
    prevent_initial_call=True
)
def explain_shap(n, img):

    # Convert stored image data back to numpy array
    arr = np.array(img, dtype=np.float32)
    
    # Get SHAP explanation
    explanation = shap_explainer.explain(arr, num_samples=1500)
    shap_values = explanation["shap_values"]
    predicted_class = explanation["predicted_class"]
    class_idx = 1 if predicted_class == "PNEUMONIA" else 0
    
    # Extract SHAP values for the relevant class (remove batch dimension)
    shap_for_class = shap_values.values[0, ..., class_idx]  # Shape: (256, 256, 3)
    
    # Convert original image to uint8
    original_img = arr.astype(np.uint8)
    
    # Create SHAP heatmap (sum across channels and normalize)
    if len(shap_for_class.shape) == 3:
        shap_heatmap = np.sum(shap_for_class, axis=-1)
    else:
        shap_heatmap = shap_for_class
    
    # Normalize to [-1, 1] range
    abs_max = np.max(np.abs(shap_heatmap))
    shap_heatmap = shap_heatmap / (abs_max + 1e-10)
    
    # Create figure with original image as base
    fig = go.Figure()
    
    # Add original image
    fig.add_trace(
        go.Image(
            z=original_img,
            hoverinfo='skip'
        )
    )
    
    # Add SHAP heatmap overlay with transparency
    fig.add_trace(
        go.Heatmap(
            z=shap_heatmap,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            opacity=0.5, 
            hoverinfo='skip',
            colorbar=dict(
                tickvals=[-1, 0, 1],
                ticktext=["Negative", "Neutral", "Positive"]
            )
        )
    )
    
    # Update layout
    fig.update_layout( 
    
        title=f"SHAP Explanation Overlay - Predicted: {predicted_class}",
        
        title_x=0.5,
        margin=dict(l=20, r=20, t=80, b=20),
        height=400,
        showlegend=False,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    
    # Adjust grid and axes
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    logger.info(f"Created SHAP overlay visualization for {predicted_class}")
    
    return dcc.Graph(
        figure=fig,
        config={'displayModeBar': False},
        style={'height': '500px'}
    ), shap_for_class.tolist()

    

@app.callback(
    Output("tab-content", "children", allow_duplicate=True),
    Output("store-gradcam", "data"),
    Input("btn-gradcam", "n_clicks"),
    State("store-img", "data"),
    prevent_initial_call=True
)
def explain_gradcam(n, img):
    arr = np.array(img, dtype=np.float32)
    res = gradcam_explainer.explain(arr)
    fig = px.imshow(res["superimposed_image"])
    fig.update_layout(title="Grad-CAM Explanation",title_x=0.5, margin=dict(l=20, r=20, t=50, b=20), coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '500px'}), res


if __name__ == "__main__":
    app.run(debug=True)
