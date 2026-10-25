"""Repair a namespace preloaded by Ubuntu's system Matplotlib .pth file.

The ROS venv inherits system packages. Ubuntu may prepopulate mpl_toolkits
with only its system directory, hiding the toolkit paired with venv Matplotlib.
Prepend the toolkit shipped with the selected Matplotlib distribution; do not
suppress missing-projection warnings.
"""

from importlib.metadata import distribution


def prepare_matplotlib_namespace():
    import mpl_toolkits
    toolkit = str(distribution('matplotlib').locate_file('mpl_toolkits'))
    mpl_toolkits.__path__ = [toolkit]+[p for p in mpl_toolkits.__path__ if p != toolkit]
