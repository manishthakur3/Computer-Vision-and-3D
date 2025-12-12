"""
Enhanced Configuration with optimized detection settings
"""
import torch

# Check for GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

CONFIG = {
    # Use YOLOv8m for balanced accuracy and speed (optimized performance)
    'yolo_model': 'yolov8m.pt',  # Balanced model: good accuracy + decent speed
    'confidence_threshold': 0.3,  # Balanced threshold for accuracy and speed
    'panel_width': 380,  # Optimized panel width
    'max_panel_objects': 12,  # Balanced object display limit
    
    # Optimized refresh intervals
    'refresh_intervals': {
        'system_status': 0.5,
        'object_info': 0.5,  # Faster updates
        'gesture_status': 0.2,
        'activity_detection': 0.3,
        'ar_interaction': 0.3
    },
    
    # Enhanced object colors
    'object_colors': {
        'person': (0, 255, 255),
        'chair': (100, 100, 255),
        'table': (100, 255, 100),
        'laptop': (255, 165, 0),
        'book': (100, 100, 255),
        'bottle': (0, 255, 0),
        'cup': (255, 255, 0),
        'cell phone': (255, 0, 255),
        'tv': (255, 99, 71),
        'keyboard': (0, 191, 255),
        'mouse': (255, 20, 147),
        'cat': (255, 140, 0),
        'dog': (0, 140, 255),
        'bird': (255, 215, 0),
        'horse': (139, 69, 19),
        'apple': (0, 255, 0),
        'banana': (0, 255, 255),
        'orange': (0, 165, 255),
        'carrot': (0, 69, 255),
        'broccoli': (0, 255, 127),
        'remote': (128, 0, 128),
        'monkey': (139, 0, 139),       
        'bear': (165, 42, 42),         
        'elephant': (169, 169, 169),   
        'zebra': (255, 255, 255),      
        'giraffe': (255, 165, 0),      
        'cow': (245, 222, 179),        
        'sheep': (255, 255, 240),      
        'potted plant': (34, 139, 34), 
        'vase': (238, 130, 238),       
        'scissors': (192, 192, 192),   
        'teddy bear': (160, 82, 45),   
        'hair drier': (128, 128, 128), 
        'toothbrush': (255, 250, 205), 
        'clock': (255, 215, 0),        
        'microwave': (105, 105, 105),  
        'oven': (139, 69, 19),         
        'sink': (70, 130, 180),        
        'refrigerator': (176, 224, 230),
        'car': (0, 0, 255),            
        'motorcycle': (0, 100, 0),     
        'bicycle': (255, 0, 0),        
        'bus': (255, 140, 0),          
        'truck': (0, 255, 127),        
        'traffic light': (0, 255, 0),  
        'fire hydrant': (255, 0, 0),   
        'stop sign': (255, 0, 0),      
        'parking meter': (192, 192, 192),
        'bench': (139, 69, 19),        
        'backpack': (255, 165, 0),     
        'umbrella': (0, 191, 255),     
        'handbag': (255, 20, 147),     
        'tie': (255, 0, 255),          
        'suitcase': (139, 69, 19),     
        'frisbee': (0, 255, 255),      
        'skis': (255, 255, 255),       
        'snowboard': (255, 255, 255),  
        'sports ball': (255, 0, 0),    
        'kite': (255, 255, 0),         
        'baseball bat': (139, 69, 19), 
        'baseball glove': (139, 69, 19),
        'skateboard': (0, 0, 0),       
        'surfboard': (0, 0, 255),      
        'tennis racket': (255, 255, 255),
        'wine glass': (255, 215, 0),   
        'knife': (192, 192, 192),      
        'bowl': (255, 215, 0),         
        'spoon': (192, 192, 192),      
        'fork': (192, 192, 192),       
        'sandwich': (210, 180, 140),   
        'hot dog': (255, 69, 0),       
        'pizza': (255, 69, 0),         
        'donut': (139, 0, 0),          
        'cake': (255, 182, 193),       
        'couch': (139, 69, 19),        
        'bed': (139, 69, 19),          
        'toilet': (255, 255, 255),     
        'tv': (0, 0, 255),             
        'laptop': (0, 0, 139),         
        'mouse': (105, 105, 105),      
        'remote': (128, 0, 128),       
        'keyboard': (0, 0, 0),         
        'microwave': (105, 105, 105),  
        'oven': (139, 69, 19),         
        'toaster': (192, 192, 192),    
        'sink': (169, 169, 169),       
        'refrigerator': (255, 255, 255),
        'book': (139, 0, 0),           
        'clock': (255, 215, 0),        
        'vase': (255, 20, 147),        
        'scissors': (192, 192, 192),   
        'teddy bear': (160, 82, 45),   
        'hair drier': (128, 128, 128), 
        'toothbrush': (255, 255, 255), 
    },
    
    # Enhanced object info database
    'object_info_db': {
        'person': {
            'name': 'Human',
            'category': 'Living Being',
            'description': 'A person detected in the scene',
            'interactions': ['wave', 'follow', 'talk'],
            'ar_model': 'human_3d',
            'details': 'Human activity monitoring enabled',
            'price': 'N/A',
            'material': 'Organic',
            'weight': '60-100 kg'
        },
        'cat': {
            'name': 'Domestic Cat',
            'category': 'Animal',
            'description': 'Common household pet, feline species',
            'interactions': ['pet', 'feed', 'play with toy'],
            'ar_model': 'cat_3d',
            'details': 'Independent and affectionate companion',
            'price': 'N/A',
            'material': 'Organic',
            'weight': '3-7 kg'
        },
        'dog': {
            'name': 'Domestic Dog',
            'category': 'Animal',
            'description': 'Loyal companion animal, canine species',
            'interactions': ['pet', 'walk', 'play fetch'],
            'ar_model': 'dog_3d',
            'details': 'Man\'s best friend, highly trainable',
            'price': 'N/A',
            'material': 'Organic',
            'weight': '5-80 kg'
        },
        'apple': {
            'name': 'Apple',
            'category': 'Fruit',
            'description': 'Crisp and juicy fruit, typically red or green',
            'interactions': ['eat', 'slice', 'juice'],
            'ar_model': 'apple_3d',
            'details': 'Rich in vitamins and fiber',
            'price': '$0.50-$2',
            'material': 'Organic',
            'weight': '0.15-0.3 kg'
        },
        'banana': {
            'name': 'Banana',
            'category': 'Fruit',
            'description': 'Tropical curved fruit with yellow peel',
            'interactions': ['peel', 'eat', 'blend'],
            'ar_model': 'banana_3d',
            'details': 'High in potassium and energy',
            'price': '$0.20-$0.60',
            'material': 'Organic',
            'weight': '0.1-0.2 kg'
        },
        'horse': {
            'name': 'Horse',
            'category': 'Animal',
            'description': 'Large domesticated mammal used for riding',
            'interactions': ['ride', 'groom', 'feed'],
            'ar_model': 'horse_3d',
            'details': 'Powerful and graceful working animal',
            'price': 'N/A',
            'material': 'Organic',
            'weight': '400-1000 kg'
        },
        'monkey': {
            'name': 'Monkey',
            'category': 'Animal',
            'description': 'Primate mammal, intelligent and agile',
            'interactions': ['observe', 'feed', 'interact'],
            'ar_model': 'monkey_3d',
            'details': 'Social animal with complex behaviors',
            'price': 'N/A',
            'material': 'Organic',
            'weight': '2-40 kg'
        },
        'cell phone': {
            'name': 'Mobile Phone',
            'category': 'Electronics',
            'description': 'Portable communication device',
            'interactions': ['call', 'text', 'browse'],
            'ar_model': 'phone_3d',
            'details': 'Smartphone with touchscreen',
            'price': '$200-$1500',
            'material': 'Glass/Metal',
            'weight': '0.15-0.3 kg'
        },
        # Add more objects as needed
    },
    
    # Detection optimization with accuracy focus
    'detection_settings': {
        'use_gpu': DEVICE == 'cuda',
        'frame_skip': 1,  # Process every frame for accuracy
        'min_object_size': 20,  # Smaller objects detected (was 40)
        'iou_threshold': 0.4,  # Slightly lower for more detections
        'max_detections': 150,  # More detections allowed
        'tracking_enabled': True,
        'tracking_max_age': 45  # Longer tracking persistence
    },
    
    'activity_thresholds': {
        'talking': 0.025,
        'laughing': 0.04,
        'eating': 0.035,
        'sleeping': 0.01,
        'reading': 0.02,
        'crying': 0.045,
        'yawning': 0.03,
        'singing': 0.028
    },
    
    'hand_landmark_colors': {
        'wrist': (255, 0, 0),
        'thumb': (0, 255, 0),
        'index': (0, 0, 255),
        'middle': (255, 255, 0),
        'ring': (255, 0, 255),
        'pinky': (0, 255, 255)
    }
}

# Enhanced target objects list with COCO dataset classes
TARGET_OBJECTS = [
    # Persons and animals
    'person', 'cat', 'dog', 'horse', 'bird', 'cow', 'sheep', 'elephant', 
    'bear', 'zebra', 'giraffe', 'monkey',
    
    # Fruits and vegetables
    'apple', 'banana', 'orange', 'carrot', 'broccoli',
    
    # Household items
    'chair', 'table', 'couch', 'bed', 'toilet',
    
    # Electronics
    'tv', 'laptop', 'cell phone', 'remote', 'keyboard', 'mouse', 
    'microwave', 'oven', 'toaster', 'refrigerator',
    
    # Kitchen items
    'bottle', 'cup', 'bowl', 'spoon', 'fork', 'knife', 'wine glass',
    
    # Food items
    'sandwich', 'hot dog', 'pizza', 'donut', 'cake',
    
    # Miscellaneous
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    
    # Vehicles
    'car', 'motorcycle', 'bicycle', 'bus', 'truck',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'
]

# Hand landmark indices with names
HAND_LANDMARKS = {
    0: ('WRIST', 'wrist'),
    1: ('THUMB_CMC', 'thumb'),
    2: ('THUMB_MCP', 'thumb'),
    3: ('THUMB_IP', 'thumb'),
    4: ('THUMB_TIP', 'thumb'),
    5: ('INDEX_MCP', 'index'),
    6: ('INDEX_PIP', 'index'),
    7: ('INDEX_DIP', 'index'),
    8: ('INDEX_TIP', 'index'),
    9: ('MIDDLE_MCP', 'middle'),
    10: ('MIDDLE_PIP', 'middle'),
    11: ('MIDDLE_DIP', 'middle'),
    12: ('MIDDLE_TIP', 'middle'),
    13: ('RING_MCP', 'ring'),
    14: ('RING_PIP', 'ring'),
    15: ('RING_DIP', 'ring'),
    16: ('RING_TIP', 'ring'),
    17: ('PINKY_MCP', 'pinky'),
    18: ('PINKY_PIP', 'pinky'),
    19: ('PINKY_DIP', 'pinky'),
    20: ('PINKY_TIP', 'pinky')
}