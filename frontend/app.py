import io, base64
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, no_update
import numpy as np
from PIL import Image
import plotly.express as px
import tensorflow as tf
import cv2
from plotly import graph_objects as go
from matplotlib import colormaps

# Your implementations
from steps.lime_explanation import LIMEExplainer
from steps.gradcam_explanation import GradCAMExplainer
from steps.gradientshap_explanation import GradientSHAPExplainer
from steps.shap_explanation import SHAPExplainer
from steps.inference import inference
from src.utils.plots import plotly_shap_image
from src.utils.configuration import ConfigurationManager
from steps.ingest_data import ingestion
from pipelines.data_pipeline import data_pipeline
from src.log import logger

# Load model and initialize explainers
model = tf.keras.models.load_model("../models/my_model_2.keras")
class_names = ["NORMAL", "PNEUMONIA"]

config = ConfigurationManager()
data_preprocess_config = config.get_data_preprocess_config()
data_inngestion_config = config.get_data_ingestion_config()


train_ds, val_ds, test_ds, index_to_class = data_pipeline(
    ingestion_config=data_inngestion_config,
    data_preprocess_config=data_preprocess_config,
)
def get_conv2d_layer_names(model):
    conv_layer_names = []

    def collect_conv_layers(m):
        for layer in m.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layer_names.append(layer.name)
            elif isinstance(layer, tf.keras.models.Model):
                collect_conv_layers(layer)

    collect_conv_layers(model)
    return conv_layer_names


conv_layer_names = get_conv2d_layer_names(model)

def get_sample(generator, sample_size=100):
    images = []
    labels = []
    while len(images) < sample_size:
        x_batch, y_batch = next(generator)
        for x, y in zip(x_batch, y_batch):
            images.append(x)
            labels.append(y)
            if len(images) >= sample_size:
                break
    return np.array(images), np.array(labels)

X_bg_large, y_bg_large = get_sample(train_ds, sample_size=200)

lime_explainer = LIMEExplainer(model)
#gradcam_explainer = GradCAMExplainer(model)

gradient_shap_explainer = GradientSHAPExplainer(
    model=model,
    background_data=X_bg_large,
    input_shape=(256, 256, 3),
    class_names=list(index_to_class.values())
)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], 
                 suppress_callback_exceptions=True)
app.title = "XAI Healthcare Dashboard"

# Enhanced Styles
UPLOAD_STYLE = {
    'width': '100%',
    'height': '180px',
    'lineHeight': '180px',
    'borderWidth': '2px',
    'borderStyle': 'dashed',
    'borderRadius': '12px',
    'textAlign': 'center',
    'margin': '15px 0',
    'cursor': 'pointer',
    'background': 'rgba(255, 255, 255, 0.7)',
    'borderColor': '#4a8bfc',
    'transition': 'all 0.3s ease',
    'hover': {
        'borderColor': '#2a6fdb',
        'backgroundColor': 'rgba(74, 139, 252, 0.1)'
    }
}

BACKGROUND_STYLE = {
    'background': 'linear-gradient(145deg, #f8fafc 0%, #e8f2fc 100%)',
    'minHeight': '100vh',
    'padding': '25px',
    'fontFamily': '"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
}

CARD_STYLE = {
    'background': 'rgba(255, 255, 255, 0.92)',
    'backdropFilter': 'blur(12px)',
    'borderRadius': '18px',
    'boxShadow': '0 10px 35px 0 rgba(31, 38, 135, 0.12)',
    'border': '1px solid rgba(255, 255, 255, 0.25)',
    'padding': '28px',
    'margin': '18px 0',
    'transition': 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
    'overflow': 'hidden',
    'position': 'relative',
    'zIndex': '1',
    ':hover': {
        'boxShadow': '0 14px 40px 0 rgba(31, 38, 135, 0.18)'
    }
}

EXPLANATION_CONTAINER_STYLE = {
    'height': '520px',
    'position': 'relative',
    'borderRadius': '12px',
    'overflow': 'hidden',
    'boxShadow': 'inset 0 0 15px rgba(0, 0, 0, 0.05)',
    'background': '#f9fbfd',
    'border': '1px solid rgba(0, 0, 0, 0.05)'
}

CONTROL_PANEL_STYLE = {
    'background': 'rgba(248, 249, 252, 0.95)',
    'borderRadius': '14px',
    'padding': '18px',
    'marginBottom': '18px',
    'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.03)',
    'border': '1px solid rgba(0, 0, 0, 0.03)'
}

# Additional modern styling elements
HEADER_STYLE = {
    'color': '#2c3e50',
    'fontWeight': '600',
    'letterSpacing': '0.5px',
    'marginBottom': '25px',
    'textShadow': '0 1px 2px rgba(0,0,0,0.05)'
}

BUTTON_STYLE = {
    'borderRadius': '8px',
    'padding': '10px 24px',
    'fontWeight': '500',
    'letterSpacing': '0.3px',
    'transition': 'all 0.2s ease',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.05)',
    ':hover': {
        'transform': 'translateY(-1px)',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'
    },
    ':active': {
        'transform': 'translateY(0)'
    }
}
app.layout = dbc.Container(
    style=BACKGROUND_STYLE,
    children=[
        # Titre principal
        dbc.Row([
            dbc.Col(html.H2("Explainable AI for Chest X-rays", 
                    style=HEADER_STYLE,
                    className="text-center my-4"))
        ]),
        
        # Nouvelle ligne pour les contrôles d'explication (horizontal)
        dbc.Row([
            # Accordéon LIME
            dbc.Col([
                dbc.Accordion([
                    dbc.AccordionItem([
                        dbc.Form([
                            dbc.Label("Number of Samples"),
                            dcc.Slider(
                                id='lime-samples',
                                min=500, max=3000, step=500,
                                value=1500,
                                marks={500: '500', 1000: '1k', 1500: '1.5k'}
                            ),
                        ]),
                        dbc.Button("Explain with LIME", id="btn-lime", 
                                  color="secondary", className="w-100 my-1", disabled=True)
                    ], title="LIME")
                ], start_collapsed=True, flush=True)
            ], width=3, style={'padding': '0 5px'}),  # Réduit la largeur et ajoute du padding
            
            # Accordéon SHAP
            dbc.Col([
                dbc.Accordion([
                    dbc.AccordionItem([
                        dbc.Form([
                            dbc.Label("Number of Samples"),
                            dcc.Slider(
                                id='shap-samples',
                                min=50, max=500, step=50,
                                value=100,
                                marks={50: '50', 100: '100', 150: '150'}
                            ),
                            dbc.Label("mask type"),
                            dcc.Dropdown(
                                id='shap-mask',
                                options=[{"label": label, "value": label}
                                        for label in ["blur(8,8)", "blur(16,16)", "blur(32,32)", 
                                                     "blur(64,64)", "blur(128,128)", "blur(256,256)"]],
                                value="blur(8,8)",
                            ),
                            dbc.Label("Opacity"),
                            dcc.Slider(
                                id='shap-opacity',
                                min=0.1, max=0.9, step=0.1,
                                value=0.6,
                                marks={0.1: '0.1', 0.5: '0.5', 0.9: '0.9'}
                            )
                        ]),
                        dbc.Button("Explain with SHAP", id="btn-shap", 
                                  color="success", className="w-100 my-1", disabled=True)
                    ], title="Partition SHAP")
                ], start_collapsed=True, flush=True)
            ], width=3, style={'padding': '0 5px'}),
            # Accordéon Grad SHAP
            dbc.Col([
                dbc.Accordion([
                    dbc.AccordionItem([
                        dbc.Form([
                            dbc.Label("Number of Samples"),
                            dcc.Slider(
                                id='grad-shap-samples',
                                min=50, max=500, step=50,
                                value=100,
                                marks={50: '50', 100: '100', 150: '150'}
                            ),
                            dbc.Label("Opacity"),
                            dcc.Slider(
                                id='grad-shap-opacity',
                                min=0.1, max=0.9, step=0.1,
                                value=0.6,
                                marks={0.1: '0.1', 0.5: '0.5', 0.9: '0.9'}
                            ),
                            dbc.Label("Colormap"),
                            dcc.Dropdown(
                                id='grad-shap-colormap',
                                options=[{"label": label, "value": value}
                                        for label, value in zip(list(colormaps), list(colormaps))],
                                value='jet'
                            )
                        ]),
                        dbc.Button("Explain with Grad SHAP", id="btn-grad-shap", 
                                  color="success", className="w-100 my-1", disabled=True)
                    ], title="Gradient SHAP")
                ], start_collapsed=True, flush=True)
            ], width=3, style={'padding': '0 5px'}),
            # Accordéon Grad-CAM
            dbc.Col([
                dbc.Accordion([
                    dbc.AccordionItem([
                        dbc.Form([
                            dbc.Label("Layer Name"),
                            dcc.Dropdown(
                                id='gradcam-layer',
                                options=[{"label": name, "value": name} for name in conv_layer_names],
                                value='block5_conv3'
                            ),
                            dbc.Label("Colormap"),
                            dcc.Dropdown(
                                id='gradcam-colormap',
                                options=[{"label": label, "value": value}
                                        for label, value in zip(list(colormaps), list(colormaps))],
                                value='jet'
                            )
                        ]),
                        dbc.Button("Explain with Grad-CAM", id="btn-gradcam", 
                                  color="danger", className="w-100 my-1", disabled=True)
                    ], title="Grad-CAM")
                ], start_collapsed=True, flush=True)
            ], width=3, style={'padding': '0 5px'})
        ], style={'margin-bottom': '20px'}),
        
        # Contenu principal (upload + visualisation)
        dbc.Row([
            # Colonne gauche (upload)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Upload Chest X-ray Image", 
                                  className="bg-primary text-white text-center"),
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
                    dbc.CardHeader("Prediction Results", 
                                  className="bg-info text-white text-center"),
                    dbc.CardBody([
                        html.Div(id="pred-info", className="h5 text-center my-2 text-dark"),
                        dbc.Progress(id="pred-confidence", className="my-3", striped=True, animated=True),
                    ])
                ], style=CARD_STYLE)
            ], width=3),
            
            # Colonne droite (visualisations)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Visual Output", className="bg-secondary text-white text-center"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("📷 Original Image", className="text-muted text-center my-2"),
                                html.Div(
                                    id="original-image",
                                    style=EXPLANATION_CONTAINER_STYLE
                                )
                            ], width=6),
                            dbc.Col([
                                html.H6("Explanation Output", className="text-muted text-center my-2"),
                                html.Div(
                                    id="tab-content",
                                    style=EXPLANATION_CONTAINER_STYLE
                                )
                            ], width=6)
                        ])
                    ])
                ], style={**CARD_STYLE, 'minHeight': '600px'})
            ], width=9)
        ]),
        
        # Stores pour les données
        dcc.Store(id="store-img"),
        dcc.Store(id="store-lime"),
        dcc.Store(id="store-shap"),
        dcc.Store(id="store-grad-shap"),
        dcc.Store(id="store-gradcam")
    ], 
    fluid=True
)
# Callback for handling image upload
@app.callback(
    Output("pred-info", "children"),
    Output("pred-confidence", "value"),
    Output("pred-confidence", "label"),
    Output("pred-confidence", "color"),
    Output("store-img", "data"),
    Output("btn-lime", "disabled"),
    Output("btn-shap", "disabled"),
    Output("btn-grad-shap", "disabled"),
    Output("btn-gradcam", "disabled"),
    Output("tab-content", "children"),
    Output("original-image", "children"),
    Input("upload", "contents")
)
def handle_upload(contents):
    if not contents:
        return no_update, no_update, no_update, no_update, None, True, True, True, True, no_update, no_update

    try:
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

        # Create original image figure with constrained size
        original_fig = px.imshow(arr.astype(np.uint8))
        original_fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0), 
            coloraxis_showscale=False,
            height=500
        )
        original_fig.update_xaxes(showticklabels=False)
        original_fig.update_yaxes(showticklabels=False)

        return (
            diagnosis_text, 
            confidence * 100, 
            f"Confidence: {score:.2f}", 
            color, 
            arr.tolist(), 
            False, 
            False, 
            False,
            False, 
            html.Div(
                "Select an explanation method from the controls", 
                className="d-flex justify-content-center align-items-center h-100",
                style={'fontSize': '1.2rem', 'color': '#6c757d'}
            ), 
            dcc.Graph(
                figure=original_fig, 
                config={'displayModeBar': False}, 
                style={'height': '100%'}
            )
        )
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return (
            "Error processing image", 
            0, 
            "Error", 
            "danger", 
            None, 
            True, 
            True, 
            True,
            True, 
            html.Div("Error displaying image", className="text-danger"), 
            html.Div("Error displaying image", className="text-danger")
        )

# Callback for LIME explanation
@app.callback(
    Output("tab-content", "children", allow_duplicate=True),
    Output("store-lime", "data"),
    Input("btn-lime", "n_clicks"),
    State("store-img", "data"),
    State("lime-samples", "value"),

    prevent_initial_call=True
)
def explain_lime(n, img, num_samples):
    if not img or not n:
        return no_update, no_update
    
    try:
        arr = np.array(img, dtype=np.float32)
        res = lime_explainer.explain(arr, num_samples=num_samples)
        
        lime_mask = res["lime_mask"]
        lime_mask_normalized = (lime_mask - lime_mask.min()) / (lime_mask.max() - lime_mask.min())
        
        heatmap = np.uint8(255 * lime_mask_normalized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        if len(arr.shape) == 2 or arr.shape[2] == 1:
            original_img = cv2.cvtColor(np.uint8(arr), cv2.COLOR_GRAY2RGB)
        else:
            original_img = np.uint8(arr)
        
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        superimposed_img = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)
        
        fig = px.imshow(superimposed_img)
        fig.update_layout(
            title="LIME Explanation",
            title_x=0.5,
            margin=dict(l=0, r=0, t=30, b=50),
            coloraxis_showscale=True,
            height=500
        )
        fig.update_coloraxes(
            colorbar=dict(
                title="Importance",
                orientation="h",
                y=-0.2,  # Position sous l'image
                xanchor="center",
                x=0.5,
                len=0.8
            )
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return (
            dcc.Graph(
                figure=fig, 
                config={'displayModeBar': False}, 
                style={'height': '100%'}
            ), 
            res
        )
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {str(e)}")
        return html.Div("Error generating LIME explanation", className="text-danger"), None

# Callback for SHAP explanation
@app.callback(
    Output("tab-content", "children", allow_duplicate=True),
    Output("store-shap", "data", allow_duplicate=True),
    Input("btn-shap", "n_clicks"),
    State("store-img", "data"),
    State("shap-samples", "value"),
    State("shap-opacity", "value"),
    State("shap-mask", "value"),
    prevent_initial_call=True
)
def explain_shap(n, img, num_samples, opacity, mask_type):
    if not img or not n:
        return no_update, no_update
    
    try:
        shap_explainer = SHAPExplainer(model, mask_type=mask_type)
        arr = np.array(img, dtype=np.float32)
        explanation = shap_explainer.explain(arr, num_samples=num_samples)
        shap_values = explanation["shap_values"]
        predicted_class = explanation["predicted_class"]
        class_idx = 1 if predicted_class == "PNEUMONIA" else 0
        
        shap_for_class = shap_values.values[0, ..., class_idx]
        original_img = arr.astype(np.uint8)
        
        fig = plotly_shap_image(
            pixel_values=np.expand_dims(original_img, axis=0),
            shap_values=np.expand_dims(shap_for_class, axis=0),
            labels=[predicted_class],
            true_labels=None,
            vmax=None,
            cmap="viridis",
            opacity=opacity
        )
        
        fig.update_layout(
            title=f"SHAP Explanation - Predicted: {predicted_class}",
            title_x=0.5,
            margin=dict(l=0, r=0, t=30, b=50),
            height=500,
            coloraxis_showscale=True,
            showlegend=False
        )
        fig.update_coloraxes(
        colorbar=dict(
            title="Importance",
            orientation="h",
            y=-0.2,  # Position sous l'image
            xanchor="center",
            x=0.5,
            len=0.8
        )
        )
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        
        return (
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': False},
                style={'height': '100%'}
            ),
            shap_for_class.tolist()
        )
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {str(e)}")
        return html.Div("Error generating SHAP explanation", className="text-danger"), None
    
# Callback for SHAP explanation
@app.callback(
    Output("tab-content", "children", allow_duplicate=True),
    Output("store-grad-shap", "data", allow_duplicate=True),
    Input("btn-grad-shap", "n_clicks"),
    State("store-img", "data"),
    State("grad-shap-samples", "value"),
    State("grad-shap-opacity", "value"),
    State("grad-shap-colormap", "value"),
    prevent_initial_call=True
)
def explain_gradient_shap(n, img, num_samples, opacity, colormap):
    if not img or not n:
        return no_update, no_update
    
    try:
        arr = np.array(img, dtype=np.float32)
        result = gradient_shap_explainer.explain(arr, nsamples=num_samples)
        shap_values = result["shap_values"]
        predicted_class = result["predicted_class"]
    
        original_img = arr.astype(np.uint8)
        
        fig = plotly_shap_image(
            pixel_values=np.expand_dims(original_img, axis=0),
            shap_values=np.squeeze(result["shap_values"], axis=-1),
            true_labels=None,
            vmax=None,
            cmap=colormap,
            opacity=opacity
        )
        
        fig.update_layout(
            title=f"Gradient SHAP Explanation - Predicted: {predicted_class}",
            title_x=0.5,
            margin=dict(l=0, r=0, t=30, b=50),
            height=500,
            coloraxis_showscale=True,
            showlegend=False
        )
        fig.update_coloraxes(
        colorbar=dict(
            title="Importance",
            orientation="h",
            y=-0.2, 
            xanchor="center",
            x=0.5,
            len=0.8
        )
        )
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        
        return (
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': False},
                style={'height': '100%'}
            ),
            result
        )
    except Exception as e:
        logger.error(f"Error generating Gradient SHAP explanation: {str(e)}")
        return html.Div("Error generating Gradient SHAP explanation", className="text-danger"), None


# Callback for Grad-CAM explanation
@app.callback(
    Output("tab-content", "children", allow_duplicate=True),
    Output("store-gradcam", "data"),
    Input("btn-gradcam", "n_clicks"),
    State("store-img", "data"),
    State("gradcam-colormap", "value"),
    State("gradcam-layer", "value"),
    prevent_initial_call=True
)
def explain_gradcam(n, img, colormap, layer_name):
    if not img or not n:
        return no_update, no_update
    
    try:
        arr = np.array(img, dtype=np.float32)
        gradcam_explainer = GradCAMExplainer(model,last_conv_layer_name=layer_name)
        res = gradcam_explainer.explain(arr, color_map=colormap)
        
        # Create figure with proper color scaling
        fig = go.Figure()
        logger.info(f"Grad-CAM heatmap shape: {res['heatmap'].shape}")
        # Add heatmap with colorbar
        fig.add_trace(go.Heatmap(
            z=res["heatmap"], 
            coloraxis="coloraxis",
            showscale=True,
            hoverinfo='none',
            opacity=0.6
        ))
        
        # Add original image
        fig.add_trace(go.Image(
            z=res["original_image"],  
            opacity=1,
            hoverinfo='none'
        ))
        
        # Update layout with colorbar settings
        fig.update_layout(
            title="Grad-CAM Explanation",
            title_x=0.5,
            margin=dict(l=0, r=0, t=50, b=80),  
            height=500,
            showlegend=False,
            coloraxis=dict(
                colorscale=colormap,
                colorbar=dict(
                    title="Activation",
                    orientation="h",
                    y=-0.2,  # Adjust based on layout
                    x=0.5,
                    xanchor="center",
                    len=0.8,
                    thickness=15
                )
            )
        )
        
        # Hide axes
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        return (
            dcc.Graph(
                figure=fig, 
                config={'displayModeBar': False}, 
                style={'height': '100%'}
            ), 
            res
        )
    except Exception as e:
        logger.error(f"Error generating Grad-CAM explanation: {str(e)}")
        return html.Div("Error generating Grad-CAM explanation", className="text-danger"), None
if __name__ == "__main__":
    app.run(debug=True)