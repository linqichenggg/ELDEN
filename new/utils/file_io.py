import os
import json
import pickle
import pandas as pd
from typing import Dict, List, Any, Optional, Union

def ensure_dir(directory: str):
    """确保目录存在
    
    Args:
        directory: 目录路径
    """
    os.makedirs(directory, exist_ok=True)

def save_json(data: Any, file_path: str, indent: int = 2, ensure_ascii: bool = False):
    """保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
        ensure_ascii: 是否确保ASCII编码
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory:
            ensure_dir(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        return True
    except Exception as e:
        print(f"保存JSON文件失败: {e}")
        return False

def load_json(file_path: str, default: Any = None) -> Any:
    """加载JSON文件
    
    Args:
        file_path: 文件路径
        default: 默认返回值（如果加载失败）
    
    Returns:
        加载的数据或默认值
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return default

def save_pickle(data: Any, file_path: str):
    """保存数据为Pickle文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory:
            ensure_dir(directory)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"保存Pickle文件失败: {e}")
        return False

def load_pickle(file_path: str, default: Any = None) -> Any:
    """加载Pickle文件
    
    Args:
        file_path: 文件路径
        default: 默认返回值（如果加载失败）
    
    Returns:
        加载的数据或默认值
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return default
    except Exception as e:
        print(f"加载Pickle文件失败: {e}")
        return default

def load_excel(file_path: str, sheet_name: Optional[Union[str, int]] = 0) -> Optional[pd.DataFrame]:
    """加载Excel文件
    
    Args:
        file_path: 文件路径
        sheet_name: 工作表名称或索引
    
    Returns:
        DataFrame对象，如果加载失败则返回None
    """
    try:
        if os.path.exists(file_path):
            return pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Excel文件不存在: {file_path}")
        return None
    except Exception as e:
        print(f"加载Excel文件失败: {e}")
        return None

def save_excel(df: pd.DataFrame, file_path: str, sheet_name: str = 'Sheet1', index: bool = False):
    """保存DataFrame为Excel文件
    
    Args:
        df: DataFrame对象
        file_path: 文件路径
        sheet_name: 工作表名称
        index: 是否包含索引
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory:
            ensure_dir(directory)
        
        df.to_excel(file_path, sheet_name=sheet_name, index=index)
        return True
    except Exception as e:
        print(f"保存Excel文件失败: {e}")
        return False

def save_csv(df: pd.DataFrame, file_path: str, index: bool = False, encoding: str = 'utf-8'):
    """保存DataFrame为CSV文件
    
    Args:
        df: DataFrame对象
        file_path: 文件路径
        index: 是否包含索引
        encoding: 文件编码
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory:
            ensure_dir(directory)
        
        df.to_csv(file_path, index=index, encoding=encoding)
        return True
    except Exception as e:
        print(f"保存CSV文件失败: {e}")
        return False

def load_csv(file_path: str, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
    """加载CSV文件
    
    Args:
        file_path: 文件路径
        encoding: 文件编码
    
    Returns:
        DataFrame对象，如果加载失败则返回None
    """
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path, encoding=encoding)
        print(f"CSV文件不存在: {file_path}")
        return None
    except Exception as e:
        print(f"加载CSV文件失败: {e}")
        return None