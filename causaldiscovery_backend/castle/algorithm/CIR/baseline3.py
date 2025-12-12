import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import argparse

# 设置中文字体支持
try:
    # 尝试使用系统可用的中文字体
    import matplotlib.font_manager as fm
    # 查找系统中的中文字体
    chinese_fonts = [f.name for f in fm.fontManager.ttflist if '\u4e00' <= f.name[0] <= '\u9fff' or 
                    'SimHei' in f.name or 'Microsoft YaHei' in f.name or 'SimSun' in f.name or 
                    'WenQuanYi' in f.name or 'Noto Sans CJK' in f.name or 'DengXian' in f.name or
                    'FangSong' in f.name or 'KaiTi' in f.name or 'STXihei' in f.name]
    
    if chinese_fonts:
        print(f"找到系统中的中文字体: {chinese_fonts}")
        matplotlib.rcParams['font.sans-serif'] = chinese_fonts + ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    else:
        # 如果没有找到中文字体，使用默认设置
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
        print("未找到系统中的中文字体，使用默认字体设置")
    
    # 解决负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置中文字体时出错: {e}")
    # 出错时使用默认设置
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 从本地gcastle库导入
from algorithm.CIR.matrix_compatibility import MatrixCompatibilityScorer
from algorithm.gcastle.trustworthyAI.gcastle.castle.common import GraphDAG
from algorithm.gcastle.trustworthyAI.gcastle.castle.metrics import MetricsDAG
from castle.algorithms import NotearsNonlinear
from algorithm.gcastle.trustworthyAI.gcastle.castle.datasets import IIDSimulation, DAG


type = 'ER'  # or `SF`
method = 'nonlinear'
sem_type = 'gp'

# 存储不同方法之间的SHD差异
method_comparison = {
    'no_prior_vs_active': [],
    'no_prior_vs_inactive': [],
    'no_prior_vs_both': [],
    'active_vs_inactive': [],
    'active_vs_both': [],
    'inactive_vs_both': []
}

# 要测试的节点数量
node_list = [10,20,40]
# 要测试的h值（边与节点的倍数）
h_list = [2,4]

# 存储x轴标签
x_labels = []


stable_eval = []
# 遍历不同的节点数量和h值
for n_nodes in node_list:
    for h in h_list:
        print(f"\n测试配置: 节点数量 = {n_nodes}, edge = {n_nodes*h}")
        x_labels.append(f"n={n_nodes},e={n_nodes*h}")
        
        # 设置随机种子
        n_edges = h * n_nodes
        weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges,
                                            weight_range=(0.5, 2.0), seed=n_edges)

        dataset = IIDSimulation(W=weighted_random_dag, n=2000,
                                method=method, sem_type=sem_type)
        true_dag, X = dataset.B, dataset.X

        # 生成一个与矩阵相同大小的随机数矩阵，范围在[0, 1)
        random_matrix = np.random.rand(*true_dag.shape)

        # 找出随机数小于采样概率的位置
        active_sampled_indices = random_matrix <= 0.4
        active_sampled_indices = active_sampled_indices.astype(int)
        # 对这些位置进行随机采样赋值
        active_matrix = true_dag * active_sampled_indices

        random_matrix = np.random.rand(*true_dag.shape)
        inactive_samples_indices = random_matrix <= 0.6
        inactive_samples_indices = inactive_samples_indices.astype(int)
        
        # 生成inactive_matrix，条件是true_dag为0且inactive_samples_indices为1的位置
        inactive_matrix = np.zeros_like(true_dag)
        inactive_positions = (true_dag == 0) & (inactive_samples_indices == 1)
        inactive_matrix[inactive_positions] = 1

        file_dir = f"algorithm/CIR/exp/compare_2/{n_nodes}_{h}"
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)

        print("没有先验因果图修正的结果")
        al = NotearsNonlinear()
        al.learn(X)
        met = MetricsDAG(al.causal_matrix, true_dag)
        print(met.metrics)

        no_prior_matrix = al.causal_matrix
        
        print("no_prior兼容性")
        result1 = MatrixCompatibilityScorer(num_subsets=n_nodes//2+8).compatibility_score(data=X, joint_matrix=no_prior_matrix)
        # print(Com)
        # print("使用先验因果图进行修正的结果 - 有向边约束")
        
         
        # print("使用先验因果图进行修正的结果 - 无边约束")
        # al2 = NotearsNonlinear(inactive_constraints=inactive_matrix)
        # al2.learn(X)
        # inactive_matrix_result = al2.causal_matrix
        
        print("使用先验因果图进行修正的结果 - 存在有边约束")
        al3 = NotearsNonlinear(active_constraints=active_matrix, active_constraint_lambda=0.01, active_method="max")
        al3.learn(X)
        both_matrix_result = al3.causal_matrix
         
        print("active兼容性")
        result2 = MatrixCompatibilityScorer(num_subsets=n_nodes//2 + 5,algo_params={"active_constraints":active_matrix, "active_constraint_lambda":0.01, "active_method":"max","true_dag":true_dag}).compatibility_score(data=X, joint_matrix=both_matrix_result)
        stable_eval.append({"node_num":n_nodes,"edge_num":n_edges,"no_prior":result1,"active":result2})
      
        
        # 计算自兼容性分数
        
        # 计算不同方法之间的SHD差异

# 绘制 stable_eval 数据的图表
if stable_eval:
        # 检查是否可能存在中文显示问题
    # 首先检查是否有环境变量或命令行参数指定语言
    force_language = os.environ.get('PLOT_LANGUAGE', '').lower()
    if force_language == 'en' or force_language == 'english':
        use_english = True
        print("根据环境变量设置，强制使用英文标签")
    elif force_language == 'zh' or force_language == 'chinese':
        use_english = False
        print("根据环境变量设置，强制使用中文标签")
    else:
        # 没有指定语言，进行自动检测
        use_english = False
        try:
            # 尝试创建一个测试图形，检查中文显示
            test_fig = plt.figure(figsize=(2, 2), dpi=72)  # 使用小尺寸以加快测试速度
            test_ax = test_fig.add_subplot(111)
            test_ax.set_title('测试中文')
            
            # 尝试使用不同的中文字体
            font_found = False
            test_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DengXian', 'KaiTi', 'FangSong', 'STXihei']
            
            for font in test_fonts:
                try:
                    # 尝试设置当前字体
                    plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                    test_ax.set_title(f'测试中文 ({font})')
                    test_fig.canvas.draw()  # 尝试绘制
                    font_found = True
                    print(f"找到可用的中文字体: {font}")
                    break
                except Exception as font_err:
                    print(f"字体 {font} 测试失败: {font_err}")
                    continue
            
            # 如果所有字体都失败，则使用英文
            if not font_found:
                print("所有中文字体测试失败，将使用英文标签")
                use_english = True
            
            # 保存测试图形并清理
            try:
                test_fig.savefig('test_font.png')
                plt.close(test_fig)
                if os.path.exists('test_font.png'):
                    os.remove('test_font.png')  # 删除测试文件
            except Exception as save_err:
                print(f"保存测试图形失败: {save_err}")
            
            if font_found:
                print("中文显示测试通过")
            
        except Exception as e:
            print(f"中文显示测试失败: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            print("将使用英文标签")
            use_english = True
    
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    x_labels = [f"n={item['node_num']},e={item['edge_num']}" for item in stable_eval]
    no_prior_values = [item['no_prior'] for item in stable_eval]
    active_values = [item['active'] for item in stable_eval]
    
    # 设置x轴位置
    x = np.arange(len(x_labels))
    
    # 根据是否使用英文设置标签
    if use_english:
        # 英文标签
        plt.plot(x, no_prior_values, 'o-', label='No Prior', color='blue', linewidth=2)
        plt.plot(x, active_values, 's-', label='Active Constraints', color='red', linewidth=2)
        plt.title('Compatibility Score Comparison', fontsize=16)
        plt.xlabel('Nodes and Edges', fontsize=14)
        plt.ylabel('Compatibility Score', fontsize=14)
    else:
        # 中文标签
        plt.plot(x, no_prior_values, 'o-', label='无先验', color='blue', linewidth=2)
        plt.plot(x, active_values, 's-', label='有向边约束', color='red', linewidth=2)
        plt.title('兼容性分数比较', fontsize=16)
        plt.xlabel('节点数和边数', fontsize=14)
        plt.ylabel('兼容性分数', fontsize=14)
    
    # 设置x轴刻度和标签
    plt.xticks(x, x_labels, rotation=45)
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    file_path = "algorithm/CIR/exp/compatibility_comparison_3.png"
    # 保存两个版本的图表
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    # 如果使用了英文，也保存一个英文版本
    if use_english:
        plt.savefig(f"algorithm/CIR/exp/compatibility_comparison_en.png", dpi=300, bbox_inches='tight')
        print("由于中文显示问题，使用了英文标签")
        print(f"图表已保存到 ${file_path} 和 compatibility_comparison_en.png")
    else:
        print(f"图表已保存到{file_path}")
    
    # 显示图表
    plt.show()
else:
    print("没有数据可供绘图")
       
