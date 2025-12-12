import os
import time
import hashlib
import logging
import random

import networkx as nx
import requests

from pyvis.network import Network
from causalnex.structure import StructureModel
from flask import Flask, render_template, request, jsonify, make_response, send_from_directory
from networkx.algorithms import dag
import shutil
from version_tree import VersionTree
import uuid  # 用于生成修改记录唯一ID
import json

# os.chdir("/Users/louhangting/Desktop/cause/project/causal_web_app-main_merged")  # 更改为你的项目路径

# 获取当前脚本文件所在的目录
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 更改当前工作目录到项目的根目录
os.chdir(current_script_dir)

SERVICE_ENDPOINT = "http://localhost:5002/causal_discovery_use_file"

VARIETY_JSON_FOLDER = "static/variety_jsons"  # 修改为存储JSON文件
os.makedirs(VARIETY_JSON_FOLDER, exist_ok=True)
VARIETY_HTML_FOLDER = "static/variety_htmls"
os.makedirs(VARIETY_HTML_FOLDER, exist_ok=True)
VERSION_CONTROL_FOLDER = "static/version_control"
os.makedirs(VERSION_CONTROL_FOLDER, exist_ok=True)
VARIETY_LAYOUT_FOLDER = "static/variety_layouts"
os.makedirs(VARIETY_LAYOUT_FOLDER, exist_ok=True)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'app.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# 添加lib目录为静态文件路由
@app.route('/lib/<path:filename>')
def lib_static(filename):
    return send_from_directory('lib', filename)

# 服务器启动时生成唯一ID，用于前端检测服务器重启
SERVER_SESSION_ID = str(uuid.uuid4())

# 删除版本的密码
DELETE_VERSION_PASSWORD = "admin123"

# -----------------------------
# 把 pending_changes 从原先的字典改成列表
pending_changes = []
# -----------------------------

def generate_pending_id():
    """生成唯一的 pending ID，使用 uuid4 或其他方式。"""
    return str(uuid.uuid4())

@app.after_request
def add_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, public, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


class CausalGraph:
    def __init__(self, json_path=None):
        self.sm = StructureModel()
        self.log = []

        if json_path:
            self.load_from_json(json_path)

    def load_from_json(self, json_path):  # 新增JSON加载方法
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            nodes = data["nodes"]
            edges = data["edges"]

            for node in nodes:
                self.add_node(node[0], type=node[1], source=node[2])
            for edge in edges:
                self.add_edge(edge[0], edge[1], relation=edge[2], source=edge[3])

    # 把当前图结构导出到 json_path
    def dump_to_json(self, json_path):
        """
        将 self.sm 中的节点/边存成和品种1.json 相同格式的 JSON：
        {
            "nodes": [ [name, type, source], ... ],
            "edges": [ [start, end, relation, source], ... ]
        }
        """
        # 收集所有节点
        node_list = []
        for n in self.sm.nodes:
            attrs = self.sm.nodes[n]
            ntype = attrs.get("type", "未知")
            nsource = attrs.get("source", "")
            node_list.append([n, ntype, nsource])

        # 收集所有边
        edge_list = []
        for (s, t) in self.sm.edges:
            ed = self.sm.get_edge_data(s, t)
            relation = ed.get("relation", "Unknown")
            source = ed.get("source", "")
            edge_list.append([s, t, relation, source])

        data = {
            "nodes": node_list,
            "edges": edge_list
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Dumped graph structure to {json_path}")

    def save_layout(self, layout_file="static/modified_layout.json", original_file="static/original_layout.json"):
        """保存当前图的布局到 JSON 文件"""
        G = nx.DiGraph(self.sm)
        # k=1.5 增加有边节点间距，scale=400 适中的整体缩放
        pos = nx.spring_layout(G, k=1.5, seed=42, scale=400, iterations=100)
        layout = {node: (float(pos[node][0]), float(pos[node][1])) for node in G.nodes}

        if not os.path.exists(original_file):  # 如果不存在原始布局文件，则保存为原始布局
            with open(original_file, 'w') as f:
                json.dump(layout, f)
            logging.info(f"原始节点布局已保存到 {original_file}")

        # 保存修改后的布局
        with open(layout_file, 'w') as f:
            json.dump(layout, f)
        logging.info(f"修改后的节点布局已保存到 {layout_file}")

    def load_layout(self, layout_file="static/layout.json"):
        """从 JSON 文件加载布局"""
        try:
            with open(layout_file, 'r') as f:
                layout = json.load(f)
            return layout
        except FileNotFoundError:
            logging.warning(f"{layout_file} 不存在，使用默认布局。")
            return {}

    def update_layout(self, layout_file="static/modified_layout.json", original_file="static/original_layout.json"):
        """更新布局时固定已有节点，仅优化新节点"""

        # 加载原始布局
        try:
            with open(original_file, 'r') as f:
                original_layout = json.load(f)
        except FileNotFoundError:
            original_layout = {}

        # 加载动态布局
        try:
            with open(layout_file, 'r') as f:
                modified_layout = json.load(f)
        except FileNotFoundError:
            modified_layout = {}

        # 合并布局：优先使用修改过的布局，否则使用原始布局
        combined_layout = {**modified_layout, **original_layout}

        G = nx.DiGraph(self.sm)
        existing_nodes = [n for n in G.nodes if n in combined_layout]
        new_nodes = [n for n in G.nodes if n not in combined_layout]

        if new_nodes:
            # 固定已有节点位置，仅优化新节点
            pos = nx.spring_layout(
                G,
                pos=combined_layout,
                fixed=existing_nodes,
                seed=42,
                scale=400,
                k=1.5,
                iterations=200
            )
            # 更新新节点坐标
            for node in new_nodes:
                combined_layout[node] = (float(pos[node][0]), float(pos[node][1]))

            # 更新修改后的布局文件
            with open(layout_file, 'w') as f:
                json.dump(combined_layout, f)
            logging.info("新增节点布局已更新")

    def add_node(self, node_name, **attrs):
        if node_name not in self.sm.nodes:
            self.sm.add_node(node_name)
            nx.set_node_attributes(self.sm, {node_name: attrs})
            logging.info(f"Added node: {node_name} with attributes {attrs}")
        else:
            logging.warning(f"Node {node_name} already exists.")

    def remove_node(self, node_name):
        if node_name in self.sm.nodes:
            self.sm.remove_node(node_name)
            logging.info(f"Removed node: {node_name}")
        else:
            logging.warning(f"Node {node_name} not found.")

    def update_node(self, node_name, **attrs):
        if node_name in self.sm.nodes:
            current_attrs = self.sm.nodes[node_name]
            for key, value in attrs.items():
                if value:
                    current_attrs[key] = value
            print('current_attrs', current_attrs)
            nx.set_node_attributes(self.sm, {node_name: current_attrs})
            logging.info(f"Updated node: {node_name} with attributes {attrs}")
        else:
            logging.warning(f"Node {node_name} not found.")

    def add_edge(self, start_node, end_node, **attrs):
        if not self.sm.has_edge(start_node, end_node):
            self.sm.add_edge(start_node, end_node)
            nx.set_edge_attributes(self.sm, {(start_node, end_node): attrs})
            logging.info(f"Added edge from {start_node} to {end_node} with attributes {attrs}")
        else:
            logging.warning(f"Edge from {start_node} to {end_node} already exists.")

    def remove_edge(self, start_node, end_node, **attrs):
        if self.sm.has_edge(start_node, end_node):
            self.sm.remove_edge(start_node, end_node)
            logging.info(f"Removed edge from {start_node} to {end_node}")
        else:
            logging.warning(f"Edge from {start_node} to {end_node} not found.")

    def update_edge(self, start_node, end_node, **attrs):
        if self.sm.has_edge(start_node, end_node):
            current_attrs = self.sm.get_edge_data(start_node, end_node)
            for key, value in attrs.items():
                if value:
                    current_attrs[key] = value
            nx.set_edge_attributes(self.sm, {(start_node, end_node): current_attrs})
            logging.info(f"Updated edge from {start_node} to {end_node} with attributes {attrs}")
        else:
            logging.warning(f"Edge from {start_node} to {end_node} not found.")

    def find_node(self, node_name):
        if node_name in self.sm.nodes:
            return self.sm.nodes[node_name]
        else:
            logging.warning(f"Node {node_name} not found.")
            return None

    def find_edge(self, start_node, end_node):
        if self.sm.has_edge(start_node, end_node):
            return self.sm.get_edge_data(start_node, end_node)
        else:
            logging.warning(f"Edge from {start_node} to {end_node} not found.")
            return None

    def visualize(self, filename="static/graph.html", layout_file="static/layout.json"):
        net = Network(notebook=False, height="600px", width="100%", directed=True)
        layout = self.load_layout(layout_file)

        # 如果布局为空或不完整，自动生成分散布局
        if not layout or not all(node in layout for node in self.sm.nodes):
            G = nx.DiGraph(self.sm)
            pos = nx.spring_layout(G, k=1.5, seed=42, scale=400, iterations=100)
            layout = {node: (float(pos[node][0]), float(pos[node][1])) for node in G.nodes}

        # 添加节点（直接使用保存的坐标）
        for node in self.sm.nodes:
            x, y = layout.get(node, (0, 0))
            net.add_node(
                node,
                label=node,
                x=x,
                y=y,
                physics=False,
                title=str(self.sm.nodes[node]),
                font={"size": 24, "face": "Arial Narrow, sans-serif"}  # 使用窄字体让英文更紧凑
            )

        for edge in self.sm.edges:
            relation_info = self.sm.get_edge_data(edge[0], edge[1])
            rel_label = relation_info.get("relation", "Unknown")
            edge_source = relation_info.get("source", "Unknown")
            edge_title = f"Relation: {rel_label}\nSource: {edge_source}"

            # 根据relation类型设置边样式
            if rel_label == "无边":
                # 无边约束：红色虚线
                net.add_edge(
                    edge[0],
                    edge[1],
                    label=rel_label,
                    arrows="to",
                    title=str(edge_title),
                    color={"color": "#e74c3c", "highlight": "#e74c3c"},
                    dashes=[5, 5],
                    width=2,
                    font={"size": 22, "face": "Arial Narrow, sans-serif", "color": "#e74c3c"}
                )
            else:
                # 普通边：实线
                net.add_edge(
                    edge[0],
                    edge[1],
                    label=rel_label,
                    arrows="to",
                    title=str(edge_title),
                    font={"size": 22, "face": "Arial Narrow, sans-serif"}
                )

        # 设置全局字体大小
        net.set_options("""{
            "physics": { "enabled": false },
            "nodes": { "font": { "size": 24, "face": "Arial Narrow, sans-serif" } },
            "edges": { "font": { "size": 22, "face": "Arial Narrow, sans-serif" } }
        }""")
        net.save_graph(filename)
        logging.info(f"Graph saved to {filename}")

    def copy_from(self, other):
        """把 other 图的结构完全复制过来"""
        self.sm.clear()
        for n, nd in other.sm.nodes(data=True):
            self.sm.add_node(n)
            nx.set_node_attributes(self.sm, {n: nd})
        for s, t, ed in other.sm.edges(data=True):
            self.sm.add_edge(s, t)
            nx.set_edge_attributes(self.sm, {(s, t): ed})


# 初始化时加载默认JSON
DEFAULT_JSON = "static/original_images/品种1.json"
cg_original = CausalGraph(DEFAULT_JSON)
cg_modified = CausalGraph()
cg_modified.copy_from(cg_original)


@app.route('/')
def main():
    # 每次打开首页时，重置可修改图与pending_changes
    global pending_changes
    pending_changes = []
    cg_modified.copy_from(cg_original)

    # 重新生成固定布局
    cg_original.save_layout("static/layout.json")

    # 生成图并使用固定布局
    cg_modified.visualize("static/modified_graph.html", "static/layout.json")
    cg_original.visualize("static/original_graph.html", "static/layout.json")
    return render_template('main.html', version=time.time())



@app.route('/list_pending_changes', methods=['GET'])
def list_pending_changes():
    """返回全部待处理修改记录"""
    return jsonify({"pending_changes": pending_changes})

@app.route('/get_nodes', methods=['GET'])
def get_nodes():
    """返回当前可修改图的所有节点列表"""
    nodes = list(cg_modified.sm.nodes())
    return jsonify({"nodes": nodes})

@app.route('/versions')
def versions():
    return render_template('versions.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/modify_graph', methods=['POST'])
def modify_graph():
    """处理对图结构的各种增删改操作，并将对应操作记录到pending_changes列表。"""
    data = request.json

    # 检查请求数据的完整性
    if not data or 'action' not in data or 'start_node' not in data:
        return jsonify({"message": "请求参数缺失或格式错误。请提供 'action' 和 'start_node' 字段。"}), 400

    action = data['action']
    s = data['start_node']
    t = data.get('end_node')
    attrs = data.get('attrs', {})
    pid = generate_pending_id()

    if action == "add_node":
        # 默认 node_type="未知", source=""；或可前端指定
        node_type = attrs.get("type", "未知")
        node_source = attrs.get("source", "")
        if s not in cg_modified.sm.nodes:
            cg_modified.add_node(s, type=node_type, source=node_source)
            cg_modified.update_layout("static/layout.json")  # 更新布局
            cg_modified.save_layout()  # 覆盖保存最新布局

            pending_changes.append({
                "id": pid,
                "change_type": "增节点",  # 改用 change_type 而不是 type
                "start": s,
                "node_attrs": {"type": node_type, "source": node_source}
            })
        else:
            return jsonify({"message": f"节点 {s} 已经存在。"}), 400  # 节点已经存在，不能重复添加



    elif action == "remove_node":
        if s in cg_modified.sm.nodes:
            old_attrs = cg_modified.sm.nodes[s].copy()
            # 记录相关边
            edges_data = []
            for pred in list(cg_modified.sm.predecessors(s)):
                ed = cg_modified.sm.get_edge_data(pred, s).copy()
                edges_data.append(("pred", pred, s, ed))
            for succ in list(cg_modified.sm.successors(s)):
                ed = cg_modified.sm.get_edge_data(s, succ).copy()
                edges_data.append(("succ", s, succ, ed))
            cg_modified.remove_node(s)
            cg_modified.update_layout("static/layout.json")  # 更新布局
            cg_modified.save_layout()
            pending_changes.append({
                "id": pid,
                "change_type": "删节点",
                "start": s,
                "node_attrs": old_attrs,
                "edges_data": edges_data
            })
        else:
            return jsonify({"message": f"节点 {s} 不存在，无法删除。"}), 404


    elif action == "update_node":
        if s in cg_modified.sm.nodes:
            old_attrs = cg_modified.sm.nodes[s].copy()
            cg_modified.update_node(s, **attrs)
            pending_changes.append({
                "id": pid,
                "change_type": "更新节点",
                "start": s,
                "old_attrs": old_attrs,
                "new_attrs": attrs
            })
        else:
            return jsonify({"message": f"节点 {s} 不存在，无法更新。"}), 404


    elif action == "add_edge":
        if not cg_modified.sm.has_edge(s, t):
            cg_modified.add_edge(s, t, **attrs)
            pending_changes.append({
                "id": pid,
                "change_type": "增边",
                "start": s,
                "end": t
            })
        else:
            return jsonify({"message": f"起点或终点节点不存在。"}), 404


    elif action == "remove_edge":
        if cg_modified.sm.has_edge(s, t):
            original_edge = cg_modified.sm.get_edge_data(s, t).copy()
            cg_modified.remove_edge(s, t)
            pending_changes.append({
                "id": pid,
                "change_type": "删边",
                "start": s,
                "end": t,
                "original_edge": original_edge
            })
        else:
            return jsonify({"message": f"边 {s} -> {t} 不存在，无法删除。"}), 404


    elif action == "update_edge":
        if cg_modified.sm.has_edge(s, t):
            old_edge_data = cg_modified.sm.get_edge_data(s, t).copy()
            cg_modified.update_edge(s, t, **attrs)
            pending_changes.append({
                "id": pid,
                "change_type": "更新边",
                "start": s,
                "end": t,
                "old_attrs": old_edge_data,
                "new_attrs": attrs
            })
        else:
            return jsonify({"message": f"边 {s} -> {t} 不存在，无法更新。"}), 404


    elif action == "change_direction":
        if cg_modified.sm.has_edge(s, t):
            old_edge_data = cg_modified.sm.get_edge_data(s, t).copy()
            cg_modified.remove_edge(s, t)
            cg_modified.add_edge(t, s, **old_edge_data)
            pending_changes.append({
                "id": pid,
                "change_type": "改方向",
                "start": s,
                "end": t,
                "original_edge": old_edge_data
            })
        else:
            return jsonify({"message": f"边 {s} -> {t} 不存在，无法改变方向。"}), 404
    else:
        return jsonify({"message": f"无效的操作类型：{action}。"}), 400

    # 修改后重新可视化
    cg_modified.visualize("static/modified_graph.html", "static/layout.json")
    return jsonify({"message": f"{action} operation.", "pending_id": pid}), 200


@app.route('/reject_change', methods=['POST'])
def reject_change():
    """根据 pending_id 回滚操作"""
    data = request.json
    pid = data.get('pending_id')
    record = next((rc for rc in pending_changes if rc["id"] == pid), None)
    if not record:
        return jsonify({"message": f"No pending change found for id={pid}"}), 404

    pending_changes.remove(record)
    rtype = record['change_type']
    s = record['start']
    t = record.get('end')

    if rtype == "增节点":
        cg_modified.remove_node(s)
    elif rtype == "删节点":
        cg_modified.add_node(s, **record['node_attrs'])
        # 恢复相关边
        for edge_info in record['edges_data']:
            cg_modified.add_edge(edge_info[1], edge_info[2], **edge_info[3])
    elif rtype == "更新节点":
        cg_modified.update_node(s, **record['old_attrs'])
    elif rtype == "增边":
        cg_modified.remove_edge(s, t)
    elif rtype == "删边":
        cg_modified.add_edge(s, t, **record['original_edge'])
    elif rtype == "更新边":
        cg_modified.update_edge(s, t, **record['old_attrs'])
    elif rtype == "改方向":
        cg_modified.remove_edge(t, s)
        cg_modified.add_edge(s, t, **record['original_edge'])

    cg_modified.visualize("static/modified_graph.html")
    return jsonify({"message": f"Rejected {rtype}, id={pid}."}), 200




@app.route('/visualize_original', methods=['GET'])
def visualize_original():
    cg_original.visualize("static/original_graph.html")
    return jsonify({"status": "success"}), 200


@app.route('/visualize_modified', methods=['GET'])
def visualize_modified():
    cg_modified.visualize("static/modified_graph.html")
    return jsonify({"status": "success"}), 200


# 使用相对路径指定品种最新版本文件夹
new_version_folder_path = "./static/variety_jsons"
# 获取文件夹中的所有文件
new_version_files = os.listdir(new_version_folder_path)

# 过滤出HTML文件并去掉扩展名
variety_list = [os.path.splitext(file)[0] for file in new_version_files if file.endswith('.json')]

# 对文件名列表进行排序
variety_list_sorted = sorted(variety_list)

variety_list = variety_list_sorted
print(variety_list)
default_variety = variety_list[0]


@app.route('/get_server_session', methods=['GET'])
def get_server_session():
    """返回服务器启动时生成的唯一ID，用于前端检测服务器重启"""
    return jsonify({"session_id": SERVER_SESSION_ID}), 200


@app.route('/get_varieties', methods=['GET'])
def get_varieties():
    return jsonify({
        "default_variety": default_variety,
        "variety_list": variety_list
    }), 200


@app.route('/regenerate_variety_html/<variety_name>', methods=['GET'])
def regenerate_variety_html(variety_name):
    """重新生成指定品种的HTML文件（自动计算分散布局）"""
    json_path = os.path.join(VARIETY_JSON_FOLDER, f"{variety_name}.json")
    if not os.path.exists(json_path):
        return jsonify({"message": f"品种 {variety_name} 不存在"}), 404
    
    target_html = os.path.join(VARIETY_HTML_FOLDER, f"{variety_name}.html")
    layout_path = os.path.join(VARIETY_LAYOUT_FOLDER, f"{variety_name}.json")
    
    v_graph = CausalGraph(json_path)
    v_graph.save_layout(layout_path)
    v_graph.visualize(target_html, layout_path)
    
    return jsonify({"message": f"已重新生成 {variety_name} 的可视化"}), 200


@app.route('/add_variety', methods=['POST'])
def add_variety():
    variety_name = request.form.get('variety_name')
    json_file = request.files.get('json_file')  # 改为接收JSON文件

    if not variety_name:
        return jsonify({"message": "品种名称不能为空"}), 400

    if variety_name in variety_list:
        return jsonify({"message": f"品种 {variety_name} 已存在"}), 400

    target_json = os.path.join(VARIETY_JSON_FOLDER, f"{variety_name}.json")
    target_html = os.path.join(VARIETY_HTML_FOLDER, f"{variety_name}.html")

    layout_path = os.path.join(VARIETY_LAYOUT_FOLDER, f"{variety_name}.json")

    if json_file:
        json_file.save(target_json)
        v_original = CausalGraph(target_json)
        v_original.save_layout(layout_path)
        v_original.visualize(target_html, layout_path)
    else:
        # 创建空JSON结构
        with open(target_json, 'w') as f:
            json.dump({"nodes": [], "edges": []}, f)
        # 创建空HTML
        with open(target_html, 'w') as f:
            f.write("<html><body><h1></h1></body></html>")
        with open(layout_path, 'w') as f:
            json.dump({}, f)

    variety_list.append(variety_name)

    # 创建版本树
    version_tree = VersionTree()

    # 修改：使用英文作为初始内容来避免中文显示问题
    vtree = VersionTree('Initial Version', variety_name=variety_name)  # 修改：使用英文替代中文
    
    # 生成初始哈希值
    timestamp_str = str(int(time.time()))
    hash_object = hashlib.sha256(timestamp_str.encode())
    root_hash = hash_object.hexdigest()[:16]

    # 设置根节点哈希值
    vtree.root.causal_graph_hash = root_hash

    # 先创建目录
    os.makedirs(f"{VERSION_CONTROL_FOLDER}/{variety_name}", exist_ok=True)

    # 修改部分开始 - 修复版本控制文件夹内的可视化问题
    version_json_path = os.path.join(VERSION_CONTROL_FOLDER, variety_name, f"{root_hash}.json")
    version_html_path = os.path.join(VERSION_CONTROL_FOLDER, variety_name, f"{root_hash}.html")
    
    # 如果有JSON数据，创建对应的JSON文件和可视化HTML
    if os.path.exists(target_json):
        # 1. 复制JSON文件到版本控制目录
        shutil.copyfile(target_json, version_json_path)
        
        # 2. 修改：根据JSON生成可视化HTML（这是关键修复）
        try:
            v_graph = CausalGraph(version_json_path)
            v_graph.visualize(version_html_path, layout_path)
        except Exception as e:
            print(f"生成初始版本可视化时出错: {str(e)}")
            # 如果可视化失败，创建一个基本的HTML文件
            with open(version_html_path, 'w', encoding='utf-8') as f:
                f.write("<html><body><h1>Initial Version</h1></body></html>")
    else:
        # 创建空JSON结构
        with open(version_json_path, 'w', encoding='utf-8') as f:
            json.dump({"nodes": [], "edges": []}, f)
        # 创建基本HTML
        with open(version_html_path, 'w', encoding='utf-8') as f:
            f.write("<html><body><h1>Initial Version</h1></body></html>")
    # 修改部分结束
    
    # 可视化版本树，保存html, json
    vtree.visualize_version_control_tree(output_file=f"{VERSION_CONTROL_FOLDER}/{variety_name}/version_tree.html")
    vtree.save_to_file(f"{VERSION_CONTROL_FOLDER}/{variety_name}/version_tree.json")

    return jsonify({"message": f"添加品种 {variety_name} 成功！"}), 200


@app.route('/upload_candidate', methods=['POST'])
def upload_candidate():
    variety_name = request.form.get('variety_name')
    json_file = request.files.get('json_file')

    if not variety_name:
        return jsonify({"message": "品种名称不能为空"}), 400

    if variety_name not in variety_list:
        return jsonify({"message": f"品种 {variety_name} 不存在"}), 404

    if not json_file:
        return jsonify({"message": "请上传候选图 JSON 文件"}), 400

    target_json = os.path.join(VARIETY_JSON_FOLDER, f"{variety_name}.json")
    target_html = os.path.join(VARIETY_HTML_FOLDER, f"{variety_name}.html")

    layout_path = os.path.join(VARIETY_LAYOUT_FOLDER, f"{variety_name}.json")

    json_file.save(target_json)

    try:
        v_graph = CausalGraph(target_json)
        v_graph.save_layout(layout_path)
        v_graph.visualize(target_html, layout_path)
    except Exception as e:
        logging.error(f"更新候选图可视化失败: {e}")
        with open(target_html, 'w', encoding='utf-8') as f:
            f.write("<html><body><h1>Candidate Upload Failed</h1></body></html>")

    os.makedirs(os.path.join(VERSION_CONTROL_FOLDER, variety_name), exist_ok=True)

    timestamp_str = str(int(time.time()))
    hash_object = hashlib.sha256(timestamp_str.encode())
    new_hash = hash_object.hexdigest()[:16]

    version_json_path = os.path.join(VERSION_CONTROL_FOLDER, variety_name, f"{new_hash}.json")
    version_html_path = os.path.join(VERSION_CONTROL_FOLDER, variety_name, f"{new_hash}.html")

    shutil.copyfile(target_json, version_json_path)

    try:
        v_graph_version = CausalGraph(version_json_path)
        v_graph_version.visualize(version_html_path, layout_path)
    except Exception as e:
        logging.error(f"生成候选图版本可视化失败: {e}")
        with open(version_html_path, 'w', encoding='utf-8') as f:
            f.write("<html><body><h1>Candidate Version</h1></body></html>")

    tree_path = f"{VERSION_CONTROL_FOLDER}/{variety_name}/version_tree.json"
    if os.path.exists(tree_path):
        vtree = VersionTree.load_from_file(tree_path)
    else:
        vtree = VersionTree('Initial Version', variety_name=variety_name)

    vtree.create_version(content='Candidate Upload', author='System', causal_graph_hash=new_hash)
    vtree.visualize_version_control_tree(output_file=f"{VERSION_CONTROL_FOLDER}/{variety_name}/version_tree.html")
    vtree.save_to_file(tree_path)

    return jsonify({"message": f"候选图已更新至品种 {variety_name}"}), 200


@app.route('/delete_variety', methods=['POST'])
def delete_variety():
    data = request.json
    v_name = data['variety_name']
    if v_name not in variety_list:
        return jsonify({"message": f"品种 {v_name} 不存在。"}), 404

    variety_list.remove(v_name)

    target_json_file = os.path.join(VARIETY_JSON_FOLDER, f"{v_name}.json")
    if os.path.exists(target_json_file):
        os.remove(target_json_file)

    target_html_file = os.path.join(VARIETY_HTML_FOLDER, f"{v_name}.html")
    if os.path.exists(target_html_file):
        os.remove(target_html_file)


    # 删除对应的版本控制文件夹
    target_folder = os.path.join(VERSION_CONTROL_FOLDER, v_name)
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    return jsonify({"message": f"已删除品种 {v_name}"}), 200


# @app.route('/visualize_variety_in_both', methods=['POST'])
# def visualize_variety_in_both():
#     data = request.json
#     if not data or 'variety_name' not in data:
#         return jsonify({"message": "请求参数缺失，请提供 'variety_name'"}), 400  # [返回码说明] 缺少品种名
#     v_name = data['variety_name'].strip()
#     json_path = os.path.join(VARIETY_JSON_FOLDER, f"{v_name}.json")
#     if not os.path.exists(json_path):
#         return jsonify({"message": f"品种 {v_name} 对应的文件不存在"}), 404  # [返回码说明] 文件未找到


#     # 重新加载原始图
#     global cg_original, cg_modified
#     cg_original = CausalGraph(json_path)
#     cg_modified = CausalGraph()
#     cg_modified.copy_from(cg_original)

#     # 生成可视化
#     cg_original.visualize("static/original_graph.html")
#     cg_modified.visualize("static/modified_graph.html")
#     return jsonify({"message": f"已加载 {v_name} 因果图"}), 200


@app.route('/visualize_variety_in_both', methods=['POST'])
def visualize_variety_in_both():
    data = request.json
    if not data or 'variety_name' not in data:
        return jsonify({"message": "请求参数缺失，请提供 'variety_name'"}), 400
    v_name = data['variety_name'].strip()
    json_path = os.path.join(VARIETY_JSON_FOLDER, f"{v_name}.json")
    if not os.path.exists(json_path):
        return jsonify({"message": f"品种 {v_name} 对应的文件不存在"}), 404

    # 1. 重置全局图对象
    global cg_original, cg_modified
    cg_original = CausalGraph(json_path)
    cg_modified = CausalGraph()
    cg_modified.copy_from(cg_original)

    # 2. 生成可视化时明确指定布局文件
    original_layout_path = os.path.join("static", "original_layout.json")
    modified_layout_path = os.path.join("static", "modified_layout.json")

    # 3. 保存原始图布局
    cg_original.save_layout(original_layout_path)
    cg_modified.save_layout(modified_layout_path)
    

    # 4. 生成可视化
    cg_original.visualize("static/original_graph.html", original_layout_path)
    cg_modified.visualize("static/modified_graph.html", modified_layout_path)

    return jsonify({"message": f"已加载 {v_name} 因果图"}), 200

@app.route('/visualize_version_in_both', methods=['POST'])
def visualize_version_in_both():
    data = request.json
    print(data)
    v_name = data['variety_name'].strip()
    version_name = data['version_name'].strip()
    version_file = data['version_path'].strip()
    print(version_file)
    start = version_file.rfind("/") + 1  # 找到最后一个 / 的位置
    end = version_file.find(".html")  # 找到 .html 的位置
    file_name = version_file[start:end]  # 截取中间部分
    print(file_name)
    
    # 加载指定版本的因果图
    version_path = os.path.join(VERSION_CONTROL_FOLDER, v_name, f"{file_name}.json")
    print(version_path)
    
     # 1. 重置全局图对象
    global cg_original, cg_modified
    cg_original = CausalGraph(version_path)
    cg_modified = CausalGraph()
    cg_modified.copy_from(cg_original)

    # 2. 生成可视化时明确指定布局文件
    original_layout_path = os.path.join("static", "original_layout.json")
    modified_layout_path = os.path.join("static", "modified_layout.json")

    # 3. 保存原始图布局
    cg_original.save_layout(original_layout_path)
    cg_modified.save_layout(modified_layout_path)
    

    # 4. 生成可视化
    cg_original.visualize("static/original_graph.html", original_layout_path)
    cg_modified.visualize("static/modified_graph.html", modified_layout_path)
    return jsonify({"message": f"已加载 {v_name} 的版本 {version_name} 因果图"})


@app.route('/cp_modified', methods=['POST'])
def cp_modified():
    data = request.json
    if not data or 'variety_name' not in data:
        return jsonify({"message": "请求参数缺失，请提供 'variety_name'"}), 400  # [返回码说明] 缺少品种名
    v_name = data['variety_name'].strip()
    json_path = os.path.join(VARIETY_JSON_FOLDER, f"{v_name}.json")
    if not os.path.exists(json_path):
        return jsonify({"message": f"品种 {v_name} 对应的文件不存在"}), 404  # [返回码说明] 文件未找到

    
    # 1. 重置全局图对象
    global cg_modified
    cg_modified = CausalGraph()
    cg_modified.copy_from(cg_original)

    # 指定源文件路径
    original_layout_path = os.path.join("static", "original_layout.json")
    modified_layout_path = os.path.join("static", "modified_layout.json")
    
    # 读取源文件内容
    with open(original_layout_path, 'r', encoding='utf-8') as source_file:
        data = json.load(source_file)

    # 写入目标文件，覆盖原有内容
    with open(modified_layout_path, 'w', encoding='utf-8') as target_file:
        json.dump(data, target_file, ensure_ascii=False, indent=4)

    # 2. 生成可视化时明确指定布局文件
    modified_layout_path = os.path.join("static", "modified_layout.json")

    # 3. 保存原始图布局
    cg_modified.save_layout(modified_layout_path)
    

    # 4. 生成可视化
    cg_modified.visualize("static/modified_graph.html", modified_layout_path)


    return jsonify({"message": f"已加载 {v_name} 因果图"}), 200


# ========== 修改：保存功能，同时保存HTML和JSON ==========
@app.route('/save_modified_graph', methods=['POST'])
def save_modified_graph():
    """
    点击“保存”时：
    1) 把右侧modified_graph.html保存到 variety_htmls/{v_name}.html
    2) 把 cg_modified 的节点、边结构dump到 variety_jsons/{v_name}.json
    """
    data = request.json
    v_name = data['variety_name'].strip()
    if v_name not in variety_list:
        return jsonify({"message": f"品种 {v_name} 不存在"}), 404

    # 1) 保存 HTML
    src_html = "static/modified_graph.html"
    target_html = os.path.join(VARIETY_HTML_FOLDER, f"{v_name}.html")
    shutil.copyfile(src_html, target_html)

    # 2) 保存 JSON (新增)
    target_json = os.path.join(VARIETY_JSON_FOLDER, f"{v_name}.json")
    cg_modified.dump_to_json(target_json)
    
    # 3) 同步更新原因果图
    global cg_original
    cg_original.copy_from(cg_modified)
    cg_original.visualize("static/original_graph.html", "static/original_layout.json")

    # 根据时间戳序列化给target_html起名字
    timestamp = int(time.time())
    # 将时间戳转换为字符串
    timestamp_str = str(timestamp)
    # 创建一个 SHA-256 哈希对象
    hash_object = hashlib.sha256(timestamp_str.encode())
    # 获取哈希值的十六进制字符串
    hash_hex = hash_object.hexdigest()[:16]
    target_html2 = os.path.join(VERSION_CONTROL_FOLDER, v_name, f"{hash_hex}.html")
    target_json2 = os.path.join(VERSION_CONTROL_FOLDER, v_name, f"{hash_hex}.json")
    layout_path = os.path.join(VARIETY_LAYOUT_FOLDER, f"{v_name}.json")
    # 保存该版本因果图到version_control文件夹
    src_json = target_json
    shutil.copyfile(src_json, target_json2)
    # 重新生成版本HTML以确保字体一致
    try:
        v_graph = CausalGraph(target_json2)
        v_graph.visualize(target_html2, layout_path)
    except Exception as e:
        logging.error(f"生成版本HTML失败: {e}")
        shutil.copyfile(src_html, target_html2)
    # 加载现在的version_tree
    # 从文件读取版本树
    vtree = VersionTree.load_from_file(f"{VERSION_CONTROL_FOLDER}/{v_name}/version_tree.json")
    # vtree.create_version(content='demo', version_number=hash_hex, author='李四')
    vtree.create_version(content='demo', author='李四', causal_graph_hash=hash_hex)
    vtree.visualize_version_control_tree(output_file=f"{VERSION_CONTROL_FOLDER}/{v_name}/version_tree.html")
    vtree.save_to_file(f"{VERSION_CONTROL_FOLDER}/{v_name}/version_tree.json")

    return jsonify({"message": f"已将修改后的因果图保存到 {v_name}.html"}), 200


# 某一品种版本树图显示功能
@app.route('/get_variety_tree', methods=['GET'])
def get_variety_tree():
    variety_name = request.args.get('variety')
    if not variety_name:
        return jsonify({"message": "品种名称不能为空"}), 400

    variety_tree_path = os.path.join("static/version_control", f"{variety_name}", "version_tree.html")
    if not os.path.exists(variety_tree_path):
        return jsonify({"message": f"品种 {variety_name} 的版本图不存在"}), 404

    return send_from_directory(variety_tree_path)


@app.route('/select_version', methods=['POST'])
def select_version():
    print("===== select_version 函数开始执行 =====")
    
    data = request.json
    print(f"步骤1: 接收到的请求数据: {data}")
    
    variety_name = data['variety_name'].strip()
    version_name = data['version_name'].strip()
    version_name = version_name.replace('(当前)', '').replace('(最新)','').strip()
    print(f"步骤2: 处理后的品种名: '{variety_name}', 版本名: '{version_name}'")
    
    # 构建版本树文件路径
    tree_json_path = f"{VERSION_CONTROL_FOLDER}/{variety_name}/version_tree.json"
    print(f"步骤3: 版本树文件路径: {tree_json_path}")
    print(f"步骤3.1: 文件是否存在: {os.path.exists(tree_json_path)}")
    
    try:
        # 加载指定品种的version_tree
        print(f"步骤4: 尝试加载版本树...")
        vtree = VersionTree.load_from_file(tree_json_path)
        print(f"步骤4.1: 版本树加载成功，根节点版本号: {vtree.root.version_number}")
        
        # 更新vtree中的self.current
        print(f"步骤5: 尝试查找版本: {version_name}")
        node = vtree.find_version(node=vtree.root, version_identifier=version_name)
        print(f"步骤5.1: 查找结果: {'找到节点' if node else '没有找到节点'}")
        
        if node is None:
            print(f"步骤5.2: 版本 {version_name} 不存在，返回404")
            return jsonify({"message": f"版本 {version_name} 不存在"}), 404
            
        # 注意：这里只是预览，不修改current，不保存版本树
        print(f"步骤6: 预览版本: {node.version_number}（不修改current）")
        
        # 获取节点的哈希值
        hash_value = node.causal_graph_hash
        if not hash_value:
            return jsonify({"message": f"版本 {version_name} 没有关联的哈希值，无法访问对应文件"}), 404

        # 使用哈希值构建文件路径
        version_json = os.path.join(VERSION_CONTROL_FOLDER, variety_name, f"{hash_value}.json")
        version_html = os.path.join(VERSION_CONTROL_FOLDER, variety_name, f"{hash_value}.html")
        layout_path = os.path.join(VARIETY_LAYOUT_FOLDER, f"{variety_name}.json")
        
        # 重新生成HTML以确保字体大小一致
        if os.path.exists(version_json):
            try:
                v_graph = CausalGraph(version_json)
                v_graph.visualize(version_html, layout_path)
            except Exception as e:
                logging.error(f"重新生成版本图失败: {e}")
        
        res_path = f"{VERSION_CONTROL_FOLDER}/{variety_name}/{hash_value}.html"
        print(f"步骤9: 返回路径: {res_path}")
        print(f"步骤9.1: 该文件是否存在: {os.path.exists(res_path)}")
        
        print("===== select_version 函数执行完毕 =====")
        return jsonify({"message": f"已切换到版本 {version_name}", "path": res_path})
        
    except Exception as e:
        print(f"错误: 在执行过程中发生异常: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print(f"错误详细信息: {traceback.format_exc()}")
        return jsonify({"message": f"切换版本失败: {str(e)}"}), 500

@app.route('/set_global_version', methods=['POST'])
def set_global_version():
    """
    确认将选中的版本设为全局因果图
    1. 更新版本树的current
    2. 将该版本的JSON复制到variety_jsons作为当前图
    3. 重新生成variety_htmls
    """
    data = request.json
    variety_name = data['variety_name'].strip()
    version_name = data['version_name'].strip()
    version_name = version_name.replace('(当前)', '').replace('(最新)', '').strip()
    
    tree_json_path = f"{VERSION_CONTROL_FOLDER}/{variety_name}/version_tree.json"
    if not os.path.exists(tree_json_path):
        return jsonify({"message": f"品种 {variety_name} 的版本树不存在"}), 404
    
    try:
        vtree = VersionTree.load_from_file(tree_json_path)
        node = vtree.find_version(node=vtree.root, version_identifier=version_name)
        
        if node is None:
            return jsonify({"message": f"版本 {version_name} 不存在"}), 404
        
        # 1. 更新current并保存版本树
        vtree.current = node
        vtree.visualize_version_control_tree(output_file=f"{VERSION_CONTROL_FOLDER}/{variety_name}/version_tree.html")
        vtree.save_to_file(tree_json_path)
        
        # 2. 将该版本的JSON复制到variety_jsons
        hash_value = node.causal_graph_hash
        if hash_value:
            version_json = os.path.join(VERSION_CONTROL_FOLDER, variety_name, f"{hash_value}.json")
            target_json = os.path.join(VARIETY_JSON_FOLDER, f"{variety_name}.json")
            target_html = os.path.join(VARIETY_HTML_FOLDER, f"{variety_name}.html")
            layout_path = os.path.join(VARIETY_LAYOUT_FOLDER, f"{variety_name}.json")
            
            if os.path.exists(version_json):
                shutil.copyfile(version_json, target_json)
                # 3. 重新生成variety_htmls
                v_graph = CausalGraph(target_json)
                v_graph.visualize(target_html, layout_path)
        
        return jsonify({"message": f"已将 {version_name} 设为全局因果图"}), 200
        
    except Exception as e:
        logging.error(f"设置全局版本失败: {e}")
        return jsonify({"message": f"设置失败: {str(e)}"}), 500


@app.route('/delete_version', methods=['POST'])
def delete_version():
    """
    删除指定版本，需要密码验证
    注意：不能删除根节点和当前全局版本
    """
    data = request.json
    variety_name = data['variety_name'].strip()
    version_name = data['version_name'].strip()
    password = data.get('password', '')
    version_name = version_name.replace('(当前)', '').replace('(最新)', '').strip()
    
    # 验证密码
    if password != DELETE_VERSION_PASSWORD:
        return jsonify({"message": "密码错误"}), 403
    
    tree_json_path = f"{VERSION_CONTROL_FOLDER}/{variety_name}/version_tree.json"
    if not os.path.exists(tree_json_path):
        return jsonify({"message": f"品种 {variety_name} 的版本树不存在"}), 404
    
    try:
        vtree = VersionTree.load_from_file(tree_json_path)
        node = vtree.find_version(node=vtree.root, version_identifier=version_name)
        
        if node is None:
            return jsonify({"message": f"版本 {version_name} 不存在"}), 404
        
        # 不能删除根节点
        if node == vtree.root:
            return jsonify({"message": "不能删除根版本"}), 400
        
        # 不能删除当前全局版本
        if node == vtree.current:
            return jsonify({"message": "不能删除当前全局版本，请先切换到其他版本"}), 400
        
        # 删除节点的JSON和HTML文件
        if node.causal_graph_hash:
            version_json = os.path.join(VERSION_CONTROL_FOLDER, variety_name, f"{node.causal_graph_hash}.json")
            version_html = os.path.join(VERSION_CONTROL_FOLDER, variety_name, f"{node.causal_graph_hash}.html")
            if os.path.exists(version_json):
                os.remove(version_json)
            if os.path.exists(version_html):
                os.remove(version_html)
        
        # 从父节点的children中移除该节点，并将子节点上移到父节点
        if node.parent:
            parent = node.parent
            # 将被删除节点的子节点移到父节点下
            for child in node.children:
                child.parent = parent
                parent.children.append(child)
            # 从父节点中移除被删除的节点
            parent.children.remove(node)
        
        # 重新保存版本树
        vtree.visualize_version_control_tree(output_file=f"{VERSION_CONTROL_FOLDER}/{variety_name}/version_tree.html")
        vtree.save_to_file(tree_json_path)
        
        return jsonify({"message": f"已删除版本 {version_name}"}), 200
        
    except Exception as e:
        logging.error(f"删除版本失败: {e}")
        return jsonify({"message": f"删除失败: {str(e)}"}), 500


@app.route('/get_current_version', methods=['GET', 'POST'])
def get_current_version():
    """
    获取指定品种的当前版本信息
    支持两种方式获取品种名:
    - POST请求中的JSON数据
    - GET请求中的URL参数
    """
    # 支持两种请求方式
    if request.method == 'POST':
        data = request.json
        if not data or 'variety_name' not in data:
            return jsonify({"message": "请求参数缺失，请提供 'variety_name'"}), 400
        variety_name = data['variety_name'].strip()
    else:  # GET请求
        variety_name = request.args.get('variety')
        if not variety_name:
            return jsonify({"message": "品种名称不能为空"}), 400
    
    # 构建版本树文件路径
    tree_json_path = f"{VERSION_CONTROL_FOLDER}/{variety_name}/version_tree.json"
    
    if not os.path.exists(tree_json_path):
        return jsonify({"message": f"品种 {variety_name} 的版本树不存在"}), 404
    
    try:
        # 加载指定品种的版本树
        vtree = VersionTree.load_from_file(tree_json_path)
        
        if not vtree.current:
            return jsonify({"message": "没有当前节点"}), 404
        
        current_version = vtree.current.version_number
        
        # 使用哈希值构建文件路径
        hash_value = vtree.current.causal_graph_hash
        if hash_value:
            # 使用版本控制文件夹中的具体版本文件
            version_json = os.path.join(VERSION_CONTROL_FOLDER, variety_name, f"{hash_value}.json")
            version_html = os.path.join(VERSION_CONTROL_FOLDER, variety_name, f"{hash_value}.html")
            layout_path = os.path.join(VARIETY_LAYOUT_FOLDER, f"{variety_name}.json")
            
            # 重新生成HTML以确保字体大小一致
            if os.path.exists(version_json):
                try:
                    v_graph = CausalGraph(version_json)
                    v_graph.visualize(version_html, layout_path)
                except Exception as e:
                    logging.error(f"重新生成版本图失败: {e}")
            
            res_path = f"/static/version_control/{variety_name}/{hash_value}.html"
        else:
            # 降级使用最新版本文件
            res_path = f"/static/variety_htmls/{variety_name}.html"
        
        return jsonify({
            "message": f"当前版本: {current_version}", 
            "current_version": current_version,
            "path": res_path,
            "hash": hash_value
        })
        
    except Exception as e:
        print(f"获取当前版本时出错: {str(e)}")
        return jsonify({"message": f"获取当前版本失败: {str(e)}"}), 500

@app.route('/get_graph', methods=['POST'])
def get_graph():
    data = dict(request.form)
    files = request.files
    if 'File' not in files:
        return jsonify({"error": "未上传文件"}), 400

    uploaded_file = request.files['File']

    if data.get('causal_recommendation') == 'yes':
        # data['nodes'] = cg_original.sm.nodes
        # data['edges'] = cg_original.sm.edges
        # print(data)
        # data = json.dumps(data, ensure_ascii=False, indent=2)
        # print(data)


        files = {
            'file': (uploaded_file.filename, uploaded_file.stream, uploaded_file.mimetype)
        }
        # 向服务端发送请求
        response = requests.post(SERVICE_ENDPOINT, data=data,files=files)
        if response.status_code == 200:
            graph_data = response.json()
            res_data = graph_data['data']
            res_data = json.loads(res_data)
            nodes = res_data['nodes']
            edges = res_data['edges']
            for node in nodes:
                cg_modified.add_node(node['node_name'], args = node['attrs'])
                # cg_modified.visualize("static/modified_graph.html")
                cg_modified.update_layout("static/layout.json")  
                cg_modified.save_layout("static/layout.json")
                # cg_modified.visualize("static/modified_graph.html")
                pid = generate_pending_id()
                pending_changes.append({
                    "id": pid,
                    "change_type": "增节点",  # 改用 change_type 而不是 type
                    "start": node['node_name'],
                    "node_attrs": {"type": node.get("type","未知"), "source": "因果推荐"}

                })
            for edge in edges:
                cg_modified.add_edge(edge['start_node'], edge['end_node'],args = edge['attrs'])
                # cg_modified.visualize("static/modified_graph.html")
                cg_modified.update_layout("static/layout.json")  
                cg_modified.save_layout("static/layout.json")
                # cg_modified.visualize("static/modified_graph.html")
                pid = generate_pending_id()
                pending_changes.append({
                    "id": pid,
                    "change_type": "增边",
                    "start": edge['start_node'],
                    "end": edge['end_node'],

                })


            cg_modified.visualize("static/modified_graph.html")
            return jsonify({"message": "成功启动因果推荐"})
        else:
            return jsonify({'error': 'Failed to fetch data from service'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    # http://127.0.0.1:5001/ 