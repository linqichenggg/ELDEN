import os
import json
import time
import hashlib
import openai
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
import time

from config_loader import get_config

class ResponseCache:
    """API响应缓存，用于避免重复请求"""
    
    def __init__(self, cache_dir="cache/api_responses"):
        """初始化缓存
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        # 内存缓存，提高读取速度
        self.memory_cache = {}
        # 缓存命中计数器
        self.hits = 0
        self.misses = 0
    
    def _get_cache_key(self, prompt: str, model: str, params: Dict) -> str:
        """生成缓存键
        
        Args:
            prompt: 提示词
            model: 模型名称
            params: 其他参数
            
        Returns:
            缓存键
        """
        # 将prompt, model和params组合生成缓存键
        params_str = json.dumps(params, sort_keys=True)
        key_str = f"{prompt}:{model}:{params_str}"
        # 使用MD5生成固定长度的键
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存文件路径
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, prompt: str, model: str, params: Dict) -> Optional[Dict]:
        """从缓存获取响应
        
        Args:
            prompt: 提示词
            model: 模型名称
            params: 其他参数
            
        Returns:
            缓存的响应，如果不存在则返回None
        """
        cache_key = self._get_cache_key(prompt, model, params)
        
        # 先检查内存缓存
        if cache_key in self.memory_cache:
            self.hits += 1
            return self.memory_cache[cache_key]
        
        # 再检查文件缓存
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    response = json.load(f)
                    # 添加到内存缓存
                    self.memory_cache[cache_key] = response
                    self.hits += 1
                    return response
            except (json.JSONDecodeError, IOError):
                # 缓存文件损坏，删除它
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        
        self.misses += 1
        return None
    
    def save(self, prompt: str, model: str, params: Dict, response: Dict):
        """保存响应到缓存
        
        Args:
            prompt: 提示词
            model: 模型名称
            params: 其他参数
            response: API响应
        """
        cache_key = self._get_cache_key(prompt, model, params)
        
        # 保存到内存缓存
        self.memory_cache[cache_key] = response
        
        # 保存到文件缓存
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"缓存保存失败: {e}")
    
    def clear(self):
        """清除所有缓存"""
        self.memory_cache.clear()
        # 删除文件缓存
        for file_name in os.listdir(self.cache_dir):
            if file_name.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, file_name))
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息
        
        Returns:
            包含命中和未命中次数的字典
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": self.hits + self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


class BatchProcessor:
    """批处理器，将多个请求批量发送"""
    
    def __init__(self, 
                 process_func: Callable, 
                 batch_size: int = 5, 
                 max_wait_time: float = 2.0):
        """初始化批处理器
        
        Args:
            process_func: 处理批量请求的函数
            batch_size: 批处理大小
            max_wait_time: 最大等待时间（秒）
        """
        self.process_func = process_func
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = queue.Queue()
        self.results = {}
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def add_request(self, request_id: str, *args, **kwargs) -> None:
        """添加请求到队列
        
        Args:
            request_id: 请求ID
            *args, **kwargs: 请求参数
        """
        self.queue.put((request_id, args, kwargs))
    
    def get_result(self, request_id: str, timeout: float = None) -> Tuple[bool, Any]:
        """获取请求结果
        
        Args:
            request_id: 请求ID
            timeout: 超时时间（秒）
            
        Returns:
            (是否成功, 结果或错误)
        """
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            with self.lock:
                if request_id in self.results:
                    result = self.results.pop(request_id)
                    return result
            time.sleep(0.1)
        return (False, "Timeout waiting for result")
    
    def _worker(self):
        """工作线程，处理批量请求"""
        batch = []
        last_process_time = time.time()
        
        while True:
            try:
                # 非阻塞式获取，允许定期检查批处理条件
                try:
                    request_id, args, kwargs = self.queue.get(block=True, timeout=0.1)
                    batch.append((request_id, args, kwargs))
                    self.queue.task_done()
                except queue.Empty:
                    pass
                
                current_time = time.time()
                # 当达到批处理大小或等待时间超过阈值时处理批量请求
                if (len(batch) >= self.batch_size or 
                    (len(batch) > 0 and current_time - last_process_time >= self.max_wait_time)):
                    self._process_batch(batch)
                    batch = []
                    last_process_time = current_time
            except Exception as e:
                print(f"批处理线程异常: {e}")
                # 避免因异常导致CPU占用率高
                time.sleep(1)
    
    def _process_batch(self, batch: List[Tuple[str, tuple, dict]]):
        """处理一批请求
        
        Args:
            batch: 请求批次，每个元素为(request_id, args, kwargs)
        """
        if not batch:
            return
        
        try:
            # 调用处理函数
            batch_results = self.process_func(batch)
            
            # 保存结果
            with self.lock:
                for request_id, result in batch_results:
                    self.results[request_id] = result
        except Exception as e:
            # 处理失败时，为每个请求设置相同的错误
            with self.lock:
                for request_id, _, _ in batch:
                    self.results[request_id] = (False, f"批处理错误: {e}")


class ApiClient:
    """API客户端，处理与LLM的交互"""
    
    def __init__(self, model=None, use_cache=True, use_batch=True):
        """初始化API客户端
        
        Args:
            model: 默认使用的模型
            use_cache: 是否使用缓存
            use_batch: 是否使用批处理
        """
        # 从配置获取API密钥和默认模型
        self.api_key = get_config("api_key", "")
        self.model = model or get_config("default_model", "gpt-3.5-turbo")
        
        # 如果API密钥存在，则设置OpenAI客户端
        if self.api_key:
            openai.api_key = self.api_key
        
        # 缓存设置
        self.use_cache = use_cache
        self.cache = ResponseCache() if use_cache else None
        
        # 批处理设置
        self.use_batch = use_batch
        self.batch_size = get_config("batch_size", 5)
        self.max_wait_time = get_config("max_wait_time", 2.0)
        if use_batch:
            self.batch_processor = BatchProcessor(
                self._process_batch, 
                batch_size=self.batch_size,
                max_wait_time=self.max_wait_time
            )
        
        # 速率限制设置
        self.rate_limit_delay = get_config("rate_limit_delay", 0.5)  # 请求之间的延迟（秒）
        self.last_request_time = 0
        
        # 并行处理设置
        self.max_workers = get_config("max_api_workers", 5)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def _respect_rate_limit(self):
        """尊重速率限制，必要时等待"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _make_api_call(self, prompt: str, model: str, **kwargs) -> Dict:
        """进行API调用
        
        Args:
            prompt: 提示词
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            API响应
        """
        self._respect_rate_limit()
        
        try:
            # 构建消息体
            messages = [{"role": "user", "content": prompt}]
            
            # 发送请求
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            return {
                "success": True,
                "model": model,
                "response": response.choices[0].message.content,
                "usage": response.usage,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _process_batch(self, batch: List[Tuple[str, tuple, dict]]) -> List[Tuple[str, Tuple[bool, Any]]]:
        """处理一批请求
        
        Args:
            batch: 请求批次，每个元素为(request_id, args, kwargs)
            
        Returns:
            结果列表，每个元素为(request_id, (success, result))
        """
        results = []
        
        for request_id, args, kwargs in batch:
            prompt = args[0]
            model = kwargs.get("model", self.model)
            params = {k: v for k, v in kwargs.items() if k != "model"}
            
            # 检查缓存
            cached_response = None
            if self.use_cache:
                cached_response = self.cache.get(prompt, model, params)
            
            if cached_response:
                results.append((request_id, (True, cached_response)))
            else:
                # 进行API调用
                response = self._make_api_call(prompt, model, **params)
                
                # 保存到缓存
                if self.use_cache and response.get("success", False):
                    self.cache.save(prompt, model, params, response)
                
                results.append((request_id, (response.get("success", False), response)))
        
        return results
    
    def generate(self, prompt: str, model: str = None, timeout: float = 30.0, **kwargs) -> Tuple[bool, Any]:
        """生成回复
        
        Args:
            prompt: 提示词
            model: 模型名称，如果为None则使用默认模型
            timeout: 超时时间（秒）
            **kwargs: 其他参数
            
        Returns:
            (是否成功, 结果或错误)
        """
        model = model or self.model
        request_id = f"{time.time()}:{hash(prompt)}"
        
        if self.use_batch:
            # 添加到批处理队列
            self.batch_processor.add_request(request_id, prompt, model=model, **kwargs)
            # 等待结果
            return self.batch_processor.get_result(request_id, timeout)
        else:
            # 直接进行API调用
            # 检查缓存
            if self.use_cache:
                cached_response = self.cache.get(prompt, model, kwargs)
                if cached_response:
                    return (True, cached_response)
            
            # 进行API调用
            response = self._make_api_call(prompt, model, **kwargs)
            
            # 保存到缓存
            if self.use_cache and response.get("success", False):
                self.cache.save(prompt, model, kwargs, response)
            
            return (response.get("success", False), response)
    
    def generate_async(self, prompt: str, model: str = None, callback: Callable = None, **kwargs):
        """异步生成回复
        
        Args:
            prompt: 提示词
            model: 模型名称，如果为None则使用默认模型
            callback: 回调函数，接收(是否成功, 结果)作为参数
            **kwargs: 其他参数
        """
        model = model or self.model
        
        def _async_generate():
            result = self.generate(prompt, model, **kwargs)
            if callback:
                callback(*result)
        
        # 提交到线程池
        self.executor.submit(_async_generate)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息
        
        Returns:
            包含命中和未命中次数的字典
        """
        if self.use_cache:
            return self.cache.get_stats()
        return {"hits": 0, "misses": 0, "total": 0, "hit_rate": 0}
    
    def clear_cache(self):
        """清除缓存"""
        if self.use_cache:
            self.cache.clear()

# 创建全局API客户端实例
api_client = ApiClient()

def get_api_client() -> ApiClient:
    """获取API客户端实例
    
    Returns:
        API客户端实例
    """
    return api_client