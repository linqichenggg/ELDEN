from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx
import random
import uuid
import math

from config_loader import get_config

class SocialNetwork:
    """社交网络，管理代理之间的关系和互动模式"""
    
    def __init__(self):
        """初始化社交网络"""
        # 创建有向图表示社交网络
        self.graph = nx.DiGraph()
        
        # 加载配置
        self.contact_rate = get_config("simulation.contact_rate", 0.3)
    
    def add_agent(self, agent_id: str, metadata: Dict[str, Any] = None):
        """添加代理到网络
        
        Args:
            agent_id: 代理ID
            metadata: 元数据
        """
        self.graph.add_node(agent_id, metadata=metadata or {})
    
    def remove_agent(self, agent_id: str):
        """从网络移除代理
        
        Args:
            agent_id: 代理ID
        """
        if self.graph.has_node(agent_id):
            self.graph.remove_node(agent_id)
    
    def add_connection(self, agent1_id: str, agent2_id: str, strength: float = 0.5, 
                       bidirectional: bool = True):
        """添加代理之间的连接
        
        Args:
            agent1_id: 代理1 ID
            agent2_id: 代理2 ID
            strength: 连接强度（0.0-1.0）
            bidirectional: 是否双向连接
        """
        if not self.graph.has_node(agent1_id) or not self.graph.has_node(agent2_id):
            return
        
        self.graph.add_edge(agent1_id, agent2_id, strength=strength)
        
        if bidirectional:
            self.graph.add_edge(agent2_id, agent1_id, strength=strength)
    
    def get_connection_strength(self, agent1_id: str, agent2_id: str) -> float:
        """获取两个代理之间的连接强度
        
        Args:
            agent1_id: 代理1 ID
            agent2_id: 代理2 ID
            
        Returns:
            连接强度（0.0-1.0），如果不存在连接则返回0.0
        """
        if self.graph.has_edge(agent1_id, agent2_id):
            return self.graph.edges[agent1_id, agent2_id].get("strength", 0.5)
        return 0.0
    
    def get_neighbors(self, agent_id: str) -> List[str]:
        """获取代理的邻居
        
        Args:
            agent_id: 代理ID
            
        Returns:
            邻居ID列表
        """
        if not self.graph.has_node(agent_id):
            return []
        
        return list(self.graph.successors(agent_id))
    
    def get_random_contact(self, agent_id: str) -> Optional[str]:
        """获取代理的随机联系人
        
        Args:
            agent_id: 代理ID
            
        Returns:
            联系人ID，如果没有联系人则返回None
        """
        neighbors = self.get_neighbors(agent_id)
        if not neighbors:
            return None
        
        # 根据连接强度加权选择
        weights = [self.get_connection_strength(agent_id, neighbor) for neighbor in neighbors]
        return random.choices(neighbors, weights=weights, k=1)[0]
    
    def generate_small_world_network(self, agents: List[str], k: int = 4, p: float = 0.1):
        """生成小世界网络
        
        Args:
            agents: 代理ID列表
            k: 每个节点的近邻数量
            p: 重连概率
        """
        # 确保所有代理已添加到网络
        for agent_id in agents:
            if not self.graph.has_node(agent_id):
                self.add_agent(agent_id)
        
        # 生成Watts-Strogatz小世界网络
        n = len(agents)
        if n <= k:
            # 如果代理数量太少，创建完全图
            for i in range(n):
                for j in range(i+1, n):
                    self.add_connection(agents[i], agents[j])
            return
        
        # 先创建环形近邻连接
        for i in range(n):
            for j in range(1, k//2 + 1):
                neighbor = (i + j) % n
                self.add_connection(agents[i], agents[neighbor])
        
        # 根据概率p重连
        for i in range(n):
            for j in range(1, k//2 + 1):
                if random.random() < p:
                    # 随机选择新邻居
                    new_neighbor = random.choice([a for a in agents if a != agents[i]])
                    
                    # 删除原连接
                    old_neighbor = agents[(i + j) % n]
                    if self.graph.has_edge(agents[i], old_neighbor):
                        self.graph.remove_edge(agents[i], old_neighbor)
                    
                    # 添加新连接
                    self.add_connection(agents[i], new_neighbor, bidirectional=False)
    
    def generate_scale_free_network(self, agents: List[str], m: int = 2):
        """生成无标度网络（Barabási–Albert模型）
        
        Args:
            agents: 代理ID列表
            m: 每个新节点连接到现有节点的边数
        """
        # 确保所有代理已添加到网络
        for agent_id in agents:
            if not self.graph.has_node(agent_id):
                self.add_agent(agent_id)
        
        n = len(agents)
        if n <= m:
            # 如果代理数量太少，创建完全图
            for i in range(n):
                for j in range(i+1, n):
                    self.add_connection(agents[i], agents[j])
            return
        
        # 创建初始完全图
        for i in range(m):
            for j in range(i+1, m):
                self.add_connection(agents[i], agents[j])
        
        # 添加剩余节点，优先连接到高度数节点
        for i in range(m, n):
            # 获取现有节点的度
            degrees = [(n, self.graph.degree(n)) for n in range(i)]
            # 计算每个节点被选择的概率（与度成正比）
            total_degree = sum(d for _, d in degrees)
            probs = [d/total_degree for _, d in degrees]
            
            # 选择m个不同的节点
            selected = set()
            while len(selected) < m:
                idx = random.choices(range(i), weights=probs, k=1)[0]
                selected.add(idx)
            
            # 连接到选中节点
            for idx in selected:
                self.add_connection(agents[i], agents[idx])
    
    def get_interaction_pairs(self) -> List[Tuple[str, str]]:
        """获取交互对，用于模拟中的代理互动
        
        Returns:
            代理对列表，每个元素为(agent1_id, agent2_id)
        """
        interaction_pairs = []
        
        # 遍历所有边
        for agent1_id, agent2_id in self.graph.edges:
            # 根据接触率决定是否交互
            if random.random() < self.contact_rate:
                interaction_pairs.append((agent1_id, agent2_id))
        
        return interaction_pairs
    
    def get_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """计算网络中心性指标
        
        Returns:
            中心性指标字典，格式为{agent_id: {"degree": 值, "betweenness": 值, ...}}
        """
        # 计算度中心性
        degree_centrality = nx.degree_centrality(self.graph)
        
        # 计算介数中心性
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        # 计算接近中心性
        closeness_centrality = nx.closeness_centrality(self.graph)
        
        # 组合结果
        result = {}
        for agent_id in self.graph.nodes:
            result[agent_id] = {
                "degree": degree_centrality.get(agent_id, 0),
                "betweenness": betweenness_centrality.get(agent_id, 0),
                "closeness": closeness_centrality.get(agent_id, 0)
            }
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """将网络转换为字典
        
        Returns:
            网络字典
        """
        # 节点
        nodes = []
        for node_id in self.graph.nodes:
            nodes.append({
                "id": node_id,
                "metadata": self.graph.nodes[node_id].get("metadata", {})
            })
        
        # 边
        edges = []
        for edge in self.graph.edges(data=True):
            from_id, to_id, data = edge
            edges.append({
                "from": from_id,
                "to": to_id,
                "strength": data.get("strength", 0.5)
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SocialNetwork':
        """从字典创建网络
        
        Args:
            data: 网络字典
            
        Returns:
            网络对象
        """
        network = cls()
        
        # 添加节点
        for node in data.get("nodes", []):
            network.add_agent(node["id"], node.get("metadata"))
        
        # 添加边
        for edge in data.get("edges", []):
            network.add_connection(
                edge["from"], 
                edge["to"], 
                edge.get("strength", 0.5),
                bidirectional=False  # 避免重复添加
            )
        
        return network

# 创建全局社交网络实例
social_network = SocialNetwork()

def get_social_network() -> SocialNetwork:
    """获取社交网络实例
    
    Returns:
        社交网络实例
    """
    return social_network