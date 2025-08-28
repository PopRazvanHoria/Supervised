# Helper to export the notebook to a .py script programmatically
import nbformat
from nbconvert import PythonExporter

NB_PATH = 'project1_vegetable_images.ipynb'
OUT_PATH = 'project1_vegetable_images.py'

nb = nbformat.read(NB_PATH, as_version=4)
exporter = PythonExporter()
source, meta = exporter.from_notebook_node(nb)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(source)
print('Exported to', OUT_PATH)
