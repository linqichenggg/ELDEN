import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from datetime import datetime
from core import RumorSimulation

class RumorSimulationExperiment:
    def __init__(self, api_key=None, model="glm-3-turbo", save_dir="experiment_results"):
        self.api_key = api_key
        self.model = model
        self.save_dir = save_dir
        
        # 确保存储目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 实验记录
        self.experiment_logs = []
        self.experiment_results = []
        
        # 输出LLM使用状态
        print(f"实验初始化完成，{'将' if api_key else '不'}使用LLM模拟用户行为")
        if api_key:
            print(f"使用模型: {model}")
        
    def run_baseline_experiment(self, initial_users=10, max_time_steps=10):
        """运行基准实验"""
        print("======= 运行基准实验 =======")
        
        # 创建模拟
        sim = RumorSimulation(
            api_key=self.api_key,
            model=self.model,
            initial_users=initial_users,
            max_time_steps=max_time_steps,
            save_dir=os.path.join(self.save_dir, "baseline")
        )
        
        # 初始化LLM客户端
        if self.api_key:
            try:
                from run import initialize_llm_client
                sim = initialize_llm_client(sim)
                print(f"成功初始化LLM客户端，使用模型: {self.model}")
            except Exception as e:
                print(f"LLM客户端初始化失败: {str(e)}")
                sim.use_llm = False
        
        # 进行模拟
        sim.simulate_message_propagation()
        
        # 记录结果
        result = {
            "experiment": "baseline",
            "result_dir": sim.save_simulation_results(),
            "users_count": len(sim.users),
            "messages_count": len(sim.messages_in_network),
            "rumor_ratio": sim.calculate_rumor_ratio()
        }
        
        self.experiment_results.append(result)
        return sim, result
    
    def run_education_experiment(self, sim_baseline, initial_users=10, max_time_steps=10):
        """运行教育水平提升实验"""
        print("======= 运行教育水平提升实验 =======")
        
        # 创建模拟
        sim = RumorSimulation(
            api_key=self.api_key,
            model=self.model,
            initial_users=initial_users,
            max_time_steps=max_time_steps,
            save_dir=os.path.join(self.save_dir, "education")
        )
        
        # 复制基准实验的用户和网络
        sim.users = {uid: user.copy() for uid, user in sim_baseline.users.items()}
        sim.social_network = sim_baseline.social_network.copy()
        
        # 提高所有用户的健康知识水平
        for user_id, user in sim.users.items():
            old_knowledge = user["health_knowledge"]
            user["health_knowledge"] = min(100, old_knowledge + 10)
            user["rumor_sensitivity"] = 100 - user["health_knowledge"]
            user["received_messages"] = []
            user["forwarded_messages"] = []
            user["verified_messages"] = []
            
            # 更新网络中的节点属性
            if user_id in sim.social_network.nodes:
                sim.social_network.nodes[user_id]["health_knowledge"] = user["health_knowledge"]
                sim.social_network.nodes[user_id]["rumor_sensitivity"] = user["rumor_sensitivity"]
        
        # 进行模拟
        sim.simulate_message_propagation()
        
        # 记录结果
        result = {
            "experiment": "education",
            "result_dir": sim.save_simulation_results(),
            "users_count": len(sim.users),
            "messages_count": len(sim.messages_in_network),
            "rumor_ratio": sim.calculate_rumor_ratio()
        }
        
        self.experiment_results.append(result)
        return sim, result
    
    def run_debunking_experiment(self, sim_baseline, initial_users=10, max_time_steps=10):
        """运行辟谣机器人实验"""
        print("======= 运行辟谣机器人实验 =======")
        
        # 创建模拟
        sim = RumorSimulation(
            api_key=self.api_key,
            model=self.model,
            initial_users=initial_users,
            max_time_steps=max_time_steps,
            save_dir=os.path.join(self.save_dir, "debunking")
        )
        
        # 复制基准实验的用户和网络
        sim.users = {uid: user.copy() for uid, user in sim_baseline.users.items()}
        sim.social_network = sim_baseline.social_network.copy()
        
        # 重置用户的消息记录
        for user_id, user in sim.users.items():
            user["received_messages"] = []
            user["forwarded_messages"] = []
            user["verified_messages"] = []
            
        # 降低辟谣机器人激活阈值，使其更容易触发
        sim.debunking_threshold = 0.1  # 10%的谣言就激活辟谣机器人
        
        # 进行模拟
        sim.simulate_message_propagation()
        
        # 记录结果
        result = {
            "experiment": "debunking",
            "result_dir": sim.save_simulation_results(),
            "users_count": len(sim.users),
            "messages_count": len(sim.messages_in_network),
            "rumor_ratio": sim.calculate_rumor_ratio(),
            "debunking_active": sim.debunking_bot_active
        }
        
        self.experiment_results.append(result)
        return sim, result
    
    def compare_experiments(self):
        """比较不同实验的结果"""
        if len(self.experiment_results) < 2:
            print("需要至少两个实验结果来进行比较")
            return
            
        # 加载各实验的消息数据
        experiment_messages = {}
        for result in self.experiment_results:
            messages_file = os.path.join(result["result_dir"], "messages.csv")
            if os.path.exists(messages_file):
                messages_df = pd.read_csv(messages_file)
                experiment_messages[result["experiment"]] = messages_df
        
        # 创建比较图表
        self._create_comparison_visualizations(experiment_messages)
        
        # 创建比较报告
        self._create_comparison_report()
        
    def _create_comparison_visualizations(self, experiment_messages):
        """创建实验比较可视化"""
        if not experiment_messages:
            return
            
        # 创建比较目录
        comparison_dir = os.path.join(self.save_dir, "comparison")
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)
        
        # 1. 比较谣言传播率
        plt.figure(figsize=(12, 6))
        
        for exp_name, messages_df in experiment_messages.items():
            # 按时间步长计算谣言比例
            time_steps = sorted(messages_df['created_at'].unique())
            rumor_ratios = []
            
            for step in time_steps:
                step_messages = messages_df[messages_df['created_at'] == step]
                if len(step_messages) > 0:
                    rumor_count = len(step_messages[step_messages['type'] == 'rumor'])
                    rumor_ratios.append(rumor_count / len(step_messages))
                else:
                    rumor_ratios.append(0)
            
            plt.plot(time_steps, rumor_ratios, marker='o', label=exp_name.capitalize())
        
        plt.xlabel('time step')
        plt.ylabel('rumor ratio')
        plt.title('rumor ratio comparison in different experiments')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(comparison_dir, 'rumor_ratio_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 比较消息传播范围
        plt.figure(figsize=(12, 6))
        
        for exp_name, messages_df in experiment_messages.items():
            # 计算谣言和官方信息的传播范围（路径长度）
            rumor_paths = messages_df[messages_df['type'] == 'rumor']['path_length']
            official_paths = messages_df[messages_df['type'] == 'official']['path_length']
            
            plt.boxplot([rumor_paths, official_paths], 
                         positions=[list(experiment_messages.keys()).index(exp_name)*3, 
                                    list(experiment_messages.keys()).index(exp_name)*3+1], 
                         widths=0.6,
                         patch_artist=True,
                         boxprops=dict(facecolor='lightblue'),
                         labels=[f"{exp_name}-谣言", f"{exp_name}-官方"])
        
        plt.xlabel('experiment and information type')
        plt.ylabel('propagation path length')
        plt.title('propagation range comparison in different experiments')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'propagation_range_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 比较验证行为
        plt.figure(figsize=(12, 6))
        
        for exp_name, messages_df in experiment_messages.items():
            # 计算谣言和官方信息的验证率
            rumor_df = messages_df[messages_df['type'] == 'rumor']
            official_df = messages_df[messages_df['type'] == 'official']
            
            rumor_verify_rate = rumor_df['verified_count'].sum() / len(rumor_df) if len(rumor_df) > 0 else 0
            official_verify_rate = official_df['verified_count'].sum() / len(official_df) if len(official_df) > 0 else 0
            
            plt.bar([exp_name + "-谣言", exp_name + "-官方"], 
                    [rumor_verify_rate, official_verify_rate],
                    alpha=0.7)
        
        plt.xlabel('experiment and information type')
        plt.ylabel('average verification rate')
        plt.title('verification behavior comparison in different experiments')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'verification_rate_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comparison_report(self):
        """创建实验比较报告"""
        comparison_dir = os.path.join(self.save_dir, "comparison")
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)
            
        with open(os.path.join(comparison_dir, 'comparison_report.txt'), 'w') as f:
            f.write("=== 实验对比报告 ===\n\n")
            
            f.write("1. 基本指标对比:\n")
            for result in self.experiment_results:
                f.write(f"\n实验: {result['experiment'].capitalize()}\n")
                f.write(f"用户数量: {result['users_count']}\n")
                f.write(f"消息总数: {result['messages_count']}\n")
                f.write(f"最终谣言比例: {result['rumor_ratio']:.4f}\n")
                
                if 'debunking_active' in result:
                    f.write(f"辟谣机器人是否激活: {'是' if result['debunking_active'] else '否'}\n")
            
            f.write("\n2. 干预效果分析:\n")
            if len(self.experiment_results) >= 3:
                baseline = next((r for r in self.experiment_results if r['experiment'] == 'baseline'), None)
                education = next((r for r in self.experiment_results if r['experiment'] == 'education'), None)
                debunking = next((r for r in self.experiment_results if r['experiment'] == 'debunking'), None)
                
                if baseline and education:
                    edu_effect = (baseline['rumor_ratio'] - education['rumor_ratio']) / baseline['rumor_ratio'] * 100
                    f.write(f"教育干预效果: 谣言比例减少了 {edu_effect:.2f}%\n")
                
                if baseline and debunking:
                    deb_effect = (baseline['rumor_ratio'] - debunking['rumor_ratio']) / baseline['rumor_ratio'] * 100
                    f.write(f"辟谣干预效果: 谣言比例减少了 {deb_effect:.2f}%\n")
                
                if education and debunking:
                    compare = (education['rumor_ratio'] - debunking['rumor_ratio']) / education['rumor_ratio'] * 100
                    if compare > 0:
                        f.write(f"辟谣比教育更有效: 多减少了 {compare:.2f}%\n")
                    else:
                        f.write(f"教育比辟谣更有效: 多减少了 {-compare:.2f}%\n")
                        
            f.write("\n3. 结论与建议:\n")
            f.write("根据实验结果，提出以下建议：\n")
            f.write("- 针对老年人群进行健康知识教育，提高其辨别谣言的能力\n")
            f.write("- 构建及时的辟谣机制，快速响应健康谣言\n")
            f.write("- 重点关注社交圈大、传播倾向高的老年人，他们是谣言传播的关键节点\n")
        
        print(f"比较报告已保存到 {os.path.join(comparison_dir, 'comparison_report.txt')}")