import sys
sys.path.append('../')
from app.utils import files
from app import settings


def list_all_models():
    
    models_available = files.get_files_matching(settings.MODELS_ROOT)

    models = []
    
    #load the models to memory only once, when the app boots
    
    for path_to_model in models_available:
        model = os.path.basename(path_to_model)
        version_id = os.path.splitext(file_name)[0]
        model = {
            "model": model,
            "version": version_id}
        models.append(model)
    return models