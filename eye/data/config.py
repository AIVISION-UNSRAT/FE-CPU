# config.py

cfg2 = {
    'name': 'CPU',
    'feature_maps': [[8, 8]],
    'min_dim': 128,
    'steps': [16],
    'min_sizes': [[16, 24, 32]],
    'aspect_ratios': [[1]],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 1.0,
    'conf_weight': 1.0,
    'gpu_train': True
}
