from world import World
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="lqc-3.24", help="Name of the run to save outputs.")
    parser.add_argument("--contact_rate", default=3, type=int, help="Contact Rate")
    parser.add_argument("--no_init_healthy", default=25, type=int, 
                        help="Number of initial healthy people in the world.")    
    parser.add_argument("--no_init_infect", default=5, type=int,
                        help="Number of initial infected people in the world.")   
    parser.add_argument("--no_days", default=10, type=int,
                        help="Total number of days the world should run.")
    parser.add_argument("--no_of_runs", default=1, type=int, 
                        help="Total number of times you want to run this code.")
    parser.add_argument("--offset", default=0, type=int, 
                        help="offset is equal to number of days if you need to load a checkpoint")
    parser.add_argument("--max_dialogue_turns", default=3, type=int,
                        help="Maximum number of turns in a dialogue")
    parser.add_argument("--dialogue_convergence", default=0.1, type=float,
                        help="Threshold for dialogue convergence")
    parser.add_argument("--save_dialogues", action="store_true",
                        help="Save dialogue data to JSON file")
    parser.add_argument("--user_data_file", default="/Users/lqcmacmini/Desktop/weibo_users.csv",
                        help="Path to the Excel file with real user data")
    parser.add_argument("--save_behaviors", action="store_true",
                        help="Save detailed agent behavior logs")
    args = parser.parse_args()

    # 创建输出目录
    for i in range(args.no_of_runs):
        output_dir = f"output/run-{i+1}"
        os.makedirs(output_dir, exist_ok=True)

    # 运行模型
    for i in range(args.no_of_runs):
        output_dir = f"output/run-{i+1}"
        
        # 初始化模型
        model = World(
            args=args,
            initial_healthy=args.no_init_healthy,
            initial_infected=args.no_init_infect,
            contact_rate=args.contact_rate
        )
        
        # 设置多轮对话参数
        model.max_dialogue_turns = args.max_dialogue_turns
        model.dialogue_convergence_threshold = args.dialogue_convergence
        
        # 运行模型
        model.run_model(checkpoint_path=output_dir, offset=args.offset)
        
        # 获取模型数据
        model_data = model.datacollector.get_model_vars_dataframe()
        
        # 确保数据行数正确
        expected_rows = args.no_days + 1  # 初始状态 + 运行天数
        if len(model_data) > expected_rows:
            print(f"WARNING: 数据行数({len(model_data)})超过预期({expected_rows})，截断数据...")
            model_data = model_data.iloc[:expected_rows]
        
        # 添加步骤列
        model_data['Step'] = range(len(model_data))
        
        # 保存数据到CSV
        model_data.to_csv(f"{output_dir}/{args.name}-data.csv")
        print(f"数据已保存到 {output_dir}/{args.name}-data.csv")
        
        # 打印最终状态
        final_state = model_data.iloc[-1]
        print("\n最终状态:")
        print(f"易感人群: {final_state['Susceptible']}")
        print(f"感染人群: {final_state['Infected']}")
        print(f"恢复人群: {final_state['Recovered']}")
        print(f"总人口: {final_state['Susceptible'] + final_state['Infected'] + final_state['Recovered']}")
        
        # 绘制SIR模型图
        plt.figure(figsize=(10, 6))
        plt.plot(model_data['Step'], model_data['Susceptible'], 'b-', label='Susceptible')
        plt.plot(model_data['Step'], model_data['Infected'], 'r-', label='Infected')
        plt.plot(model_data['Step'], model_data['Recovered'], 'g-', label='Recovered')
        plt.xlabel('step')
        plt.ylabel('number')
        plt.title('SIR model')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{args.name}-sir.png")
        plt.close()
        
        # 保存对话数据
        if args.save_dialogues:
            model.save_dialogue_data(f"{output_dir}/{args.name}-dialogues.json")
        
        # 保存代理人信念变化
        agent_beliefs = {
            "agents": []
        }
        
        for agent in model.schedule.agents:
            agent_data = {
                "id": agent.unique_id,
                "name": agent.name,
                "age": agent.age,
                "initial_health": "Infected" if agent.unique_id >= args.no_init_healthy else "Susceptible",
                "final_health": agent.health_condition,
                "belief_history": agent.beliefs,
                "opinion_history": agent.opinions,
                "dialogue_partners": agent.dialogue_partners
            }
            agent_beliefs["agents"].append(agent_data)
        
        with open(f"{output_dir}/{args.name}-agent-beliefs.json", "w", encoding="utf-8") as f:
            json.dump(agent_beliefs, f, ensure_ascii=False, indent=2)
        
        print(f"代理人信念数据已保存到 {output_dir}/{args.name}-agent-beliefs.json")
        
        # 在模型运行完成后保存行为日志
        if args.save_behaviors:
            behavior_path = f"{output_dir}/{args.name}-behaviors.json"
            model.save_agent_behavior_logs(behavior_path)
            print(f"行为日志已保存到 {behavior_path}")