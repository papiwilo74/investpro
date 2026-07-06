import numpy as np
import pandas as pd
import math

def sanitize_for_json(obj):
    """
    Convierte recursivamente tipos de numpy/pandas y NaN/Inf a tipos nativos
    de Python que sean perfectamente compatibles con JSON.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, (tuple, set)):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, (pd.Timestamp, pd.Period)):
        return obj.strftime("%Y-%m-%d")
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    elif isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    elif pd.isna(obj):
        return None
    return obj
