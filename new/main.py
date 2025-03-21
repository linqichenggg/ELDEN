import os
import argparse
import time
import matplotlib.pyplot as plt
import pandas as pd

from config_loader import load_config, get_config
from modules.world import create_world, World
from utils.file_io import ensure_dir

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="健康谣言传播多代理模拟")
    
    # 基本参数
    parser.add_argument("--name", type=str, default=None, help="模拟名称")
    parser.add_argument("--contact_rate", type=float, default=None, help="接触率")
    parser.add_argument("--no_init_healthy", type=int, default=None, help="初始健康人数")
    parser.add_argument("--no_init_infect", type=int, default=None, help="初始感染人数")
    parser.add_argument("--no_days", type=int, default=None, help="模拟天数")
    
    # 多次运行参数
    parser.add_argument("--no_of_runs", type=int, default=1, help="运行次数")
    parser.add_argument("--offset", type=int, default=0, help="运行偏移量")
    
    # 文件参数
    parser.add_argument("--user_data_file", type=str, default=None, help="用户数据文件路径")
    parser.add_argument("--config_file", type=str, default=None, help="配置文件路径")
    parser.add_argument("--load_from_run", type=str, default=None, help="从指定运行加载检查点")
    
    # 功能开关
    parser.add_argument("--save_dialogues", action="store_true", help="保存对话")
    parser.add_argument("--save_behaviors", action="store_true", help="保存行为日志")
    parser.add_argument("--no_checkpoint", action="store_true", help="不保存检查点")
    
    return parser.parse_args()

def setup_directories():
    """设置目录"""
    ensure_dir("output")
    ensure_dir("checkpoint")
    ensure_dir("data")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置目录
    setup_directories()
    
    # 加载配置
    load_config(args.config_file, args)
    
    # 获取配置参数
    name = get_config("simulation.name", "健康谣言传播模拟")
    initial_healthy = get_config("simulation.initial_healthy", 100)
    initial_infected = get_config("simulation.initial_infected", 5)
    contact_rate = get_config("simulation.contact_rate", 0.3)
    days = get_config("simulation.days", 30)
    no_of_runs = args.no_of_runs
    offset = args.offset
    user_data_file = get_config("paths.user_data_file", args.user_data_file)
    
    # 显示配置信息
    print(f"模拟配置:")
    print(f"  名称: {name}")
    print(f"  初始健康人数: {initial_healthy}")
    print(f"  初始感染人数: {initial_infected}")
    print(f"  接触率: {contact_rate}")
    print(f"  模拟天数: {days}")
    print(f"  运行次数: {no_of_runs}")
    if user_data_file:
        print(f"  用户数据文件: {user_data_file}")
    
    # 多次运行
    for run in range(offset, offset + no_of_runs):
        run_name = f"{name}-{run+1}"
        print(f"\n开始运行 {run+1}/{no_of_runs}: {run_name}")
        
        # 创建或加载世界模型
        if args.load_from_run:
            checkpoint_path = os.path.join("checkpoint", f"run-{args.load_from_run}", f"{name}-checkpoint.pkl")
            if os.path.exists(checkpoint_path):
                world = World.load_checkpoint(checkpoint_path)
                if world is None:
                    print(f"加载检查点失败，创建新模型")
                    world = create_world(
                        initial_healthy=initial_healthy,
                        initial_infected=initial_infected,
                        contact_rate=contact_rate,
                        name=run_name,
                        days=days,
                        user_data_file=user_data_file
                    )
            else:
                print(f"检查点不存在，创建新模型")
                world = create_world(
                    initial_healthy=initial_healthy,
                    initial_infected=initial_infected,
                    contact_rate=contact_rate,
                    name=run_name,
                    days=days,
                    user_data_file=user_data_file
                )
        else:
            world = create_world(
                initial_healthy=initial_healthy,
                initial_infected=initial_infected,
                contact_rate=contact_rate,
                name=run_name,
                days=days,
                user_data_file=user_data_file
            )
        
        # 运行模拟
        start_time = time.time()
        world.run()
        end_time = time.time()
        print(f"运行耗时: {end_time - start_time:.2f}秒")
        
        # 保存结果
        world.save_results()
        
        # 保存检查点
        if not args.no_checkpoint:
            world.save_checkpoint()
        
        # 绘制SIR曲线
        try:
            df = pd.read_csv(os.path.join(world.data_collector.run_dir, f"{world.name}-data.csv"))
            
            plt.figure(figsize=(10, 6))
            plt.plot(df['S'], label='易感染')
            plt.plot(df['I'], label='感染')
            plt.plot(df['R'], label='恢复')
            plt.title(f'SIR模型 - {run_name}')
            plt.xlabel('时间步')
            plt.ylabel('人数')
            plt.legend()
            
            # 保存图表
            plt.savefig(os.path.join(world.data_collector.run_dir, f"{world.name}-sir.png"))
            plt.close()
        except Exception as e:
            print(f"绘制SIR曲线失败: {e}")
    
    print("\n所有运行完成")

if __name__ == "__main__":
    main()