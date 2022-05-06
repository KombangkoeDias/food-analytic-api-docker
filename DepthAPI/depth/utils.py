import io
import base64
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def visualize_depth(depth, figsize=(6, 5), norm_value=(1, 4), cmap="jet"):
    plt.figure(figsize=figsize)
    norm = colors.Normalize(norm_value[0], norm_value[1])
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    plt.imshow(depth, cmap=cmap, norm=norm)
    plt.axis("off")
    plt.colorbar(scalar)
    io_bytes = io.BytesIO()
    plt.savefig(io_bytes, format="png")
    io_bytes.seek(0)
    base_64_image = base64.encodebytes(io_bytes.getvalue()).decode("ascii")
    return base_64_image
