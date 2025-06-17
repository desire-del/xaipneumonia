import plotly.graph_objects as go
import numpy as np
from matplotlib import colormaps

def plotly_shap_image(
    shap_values: list[np.ndarray] | np.ndarray,
    pixel_values: np.ndarray,
    labels: list[str] = None,
    true_labels: list[str] = None,
    cmap: str = "RdBu",
    vmax: float = None,
    opacity: float = 0.6
):
    if isinstance(shap_values, np.ndarray):
        shap_values = [shap_values]  # force to list

    n_images = pixel_values.shape[0]
    n_outputs = len(shap_values)

    figs = []  # stocke toutes les figures

    for row in range(n_images):
        fig = go.Figure()
        img = pixel_values[row]

        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)

        if img.shape[2] == 1:
            base_image = img.squeeze()
            if base_image.dtype != np.uint8:
                base_image = base_image.astype(np.uint8)
            fig.add_trace(go.Image(z=base_image, colormodel="gray"))
            gray_image = base_image
        else:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            fig.add_trace(go.Image(z=img))
            gray_image = (0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2])

        for i in range(n_outputs):
            sv = shap_values[i][row]
            if sv.shape[-1] > 1:
                sv = sv.sum(-1)

            vmax_val = vmax if vmax is not None else np.nanpercentile(np.abs(sv), 99.9)

            cmap_fn = colormaps.get_cmap(cmap)
            rgba = cmap_fn((sv + vmax_val) / (2 * vmax_val))  # Normalize to [0,1]
            rgba_img = (rgba[:, :, :3] * 255).astype(np.uint8)

            fig.add_trace(go.Image(z=rgba_img, opacity=opacity, colormodel="rgb"))

        title_parts = []
        if true_labels:
            title_parts.append(f"True: {true_labels[row]}")
        if labels:
            if isinstance(labels[row], list):
                title_parts.append(f"Pred: {', '.join(labels[row])}")
            else:
                title_parts.append(f"Pred: {labels[row]}")
        title = " | ".join(title_parts)

        fig.update_layout(
            title=title,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis_showticklabels=False,
            yaxis_showticklabels=False,
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        figs.append(fig)

    return figs if len(figs) > 1 else figs[0]
