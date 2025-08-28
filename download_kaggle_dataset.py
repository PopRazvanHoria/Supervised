import os
import traceback

DATASET_ID = "misrakahmed/vegetable-image-dataset"

try:
    import kagglehub
    print('kagglehub version:', getattr(kagglehub, '__version__', 'unknown'))
    path = kagglehub.dataset_download(DATASET_ID)
    print('Path to dataset files:', path)
    # If path is a file (zip), extract to ./data if possible
    if os.path.isfile(path):
        from zipfile import ZipFile
        outdir = os.path.join(os.path.abspath('.'), 'data')
        os.makedirs(outdir, exist_ok=True)
        print('Extracting to', outdir)
        with ZipFile(path, 'r') as z:
            z.extractall(outdir)
        print('Extraction complete:', outdir)
except Exception as e:
    print('Download failed:')
    traceback.print_exc()
    print('\nCommon fixes: ensure Kaggle credentials are available, or download dataset manually and set DATA_DIR in the notebook.')
