import os.path
import sys
vit_module_path = os.path.join(os.path.dirname(__file__), 'DETR')
if vit_module_path not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DETR'))
# from DETR import detr_export_model
""" Model Visualize """
# if __name__ == '__main__':
#     detr_export_model()
