import json
import os
import time
import logging
import traceback

import networkx as nx
from pyvis.network import Network
from causalnex.structure import StructureModel
from flask import Flask, render_template, request, jsonify, make_response, send_from_directory, app, after_this_request
from networkx.algorithms import dag
from pathlib import Path

from statsmodels.treatment.treatment_effects import ate_ipw
from torch.fx.experimental.unification.multipledispatch.dispatcher import source
from werkzeug.utils import secure_filename

from causaldiscovery_backend.util.discover_pure_analysis import Causal_Discovery

# 获取当前目录的父目录的父目录
root_dir = Path.cwd()

# 转换为字符串（可选）

# 更改当前工作目录到项目的根目录

os.chdir(root_dir.resolve())
log_dir = "causaldiscovery_backend/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'app.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # 禁用缓存
app.config['UPLOAD_FOLDER'] = 'causaldiscovery_backend/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

causal_discovery = Causal_Discovery()
sm = None
# 禁止缓存的响应头
@app.after_request
def add_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, public, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response







@app.route("/causal_discovery_use_file", methods=['POST'])
def causal_discovery_use_file():
    try:
        # 检查文件是否上传

        if 'file' not in request.files:
            return jsonify({"error": "未上传文件"}), 400
        file = request.files['file']
        data = dict(request.form)
        target = None
        # print(file)
        # print(data)
        attributes = data.get('featureQuality')
        if  (attributes == ''  or attributes is None):
            target = None
        else:
            target = attributes
        # 检查文件名是否有效
        if file.filename == '':
            return jsonify({"error": "无效文件名"}), 400
        method = data.get('Algorithm')
        if not method:
            return jsonify({"error": "未提供method字段"}), 400


        # 检查必填表单字段（特征品质和品种）
        # feature_quality = request.form.get('feature_quality')
        # category = request.form.get('category')
        # if not feature_quality or not category:
        #     return jsonify({"error": "缺少特征品质或品种参数"}), 400

        # 安全保存文件
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        sm = causal_discovery.discover_file(file.stream,target= target,use_model=method.lower())
        edges = []


        for out_node,item in sm._adj.items():
            for in_node,attrs in item.items():
                if attrs["mean_effect"] == 0:
                    relation = "unknown"
                elif attrs["mean_effect"] > 0:
                    relation = "positive"
                else:
                    relation = "negative"
                attrs["source"] = f"因果发现算法"
                attrs["relation"] = relation
                edges.append({"start_node": out_node,"end_node":in_node,
                              "attrs":attrs})



        nodes = []


        for node in sm.nodes:
            nodes.append({
                "node_name": node,
                "attrs": {"bias":sm._node[node]["bias"]}

            })

        js = {"nodes":nodes,"edges":edges}
        # 返回处理结果（可扩展为保存到数据库）
        return jsonify({
            "status": "success",
            "code":200,
            "data":json.dumps(js),
            "message": f"{filename}发现success",
            "page":  render_template("network.html",name="causal_discovery")
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e),"code":500})


@app.route("/predict_target_use_current_graph", methods=['GET', 'POST'])
def predict_target_use_current_graph():
    pass



#
#
# @app.route('/')
# def index():
#     # 返回渲染的模板，并添加版本号
#     return render_template('index.html', version=time.time())
#
#
# # 节点操作
# @app.route('/add_node', methods=['POST'])
# def add_node():
#     data = request.json
#     node_name = data['node_name']
#     attrs = data.get('attrs', {})
#     # 添加默认属性
#     if 'type' not in attrs:
#         attrs['type'] = 'Observable'
#     if 'source' not in attrs:
#         attrs['source'] = 'Unknown'
#     cg.add_node(node_name, **attrs)
#     cg.visualize()
#     return jsonify({"status": "success", "message": f"Node {node_name} added."})
#
#
# @app.route('/update_node', methods=['POST'])
# def update_node():
#     data = request.json
#     node_name = data['node_name']
#     attrs = data.get('attrs', {})
#     cg.update_node(node_name, **attrs)
#     cg.visualize()
#     return jsonify({"status": "success", "message": f"Node {node_name} updated."})
#
#
# # 边操作
# @app.route('/add_edge', methods=['POST'])
# def add_edge():
#     data = request.json
#     start_node = data['start_node']
#     end_node = data['end_node']
#     attrs = data.get('attrs', {})
#     # 添加默认属性
#     if 'relation' not in attrs:
#         attrs['relation'] = 'Unknown'
#     if 'source' not in attrs:
#         attrs['source'] = 'Unknown'
#     cg.add_edge(start_node, end_node, **attrs)
#     cg.visualize()
#     return jsonify({"status": "success", "message": f"Edge from {start_node} to {end_node} added."})

#
# @app.route('/update_edge', methods=['POST'])
# def update_edge():
#     data = request.json
#     start_node = data['start_node']
#     end_node = data['end_node']
#     attrs = data.get('attrs', {})
#     cg.update_edge(start_node, end_node, **attrs)
#     cg.visualize()
#     return jsonify({"status": "success", "message": f"Edge from {start_node} to {end_node} updated."})
#
#
# # 删除操作
# @app.route('/remove_node', methods=['POST'])
# def remove_node():
#     data = request.json
#     node_name = data['node_name']
#     cg.remove_node(node_name)
#     cg.visualize()  # 重新生成可视化文件
#     return jsonify({"status": "success", "message": f"Node {node_name} removed."})
#
#
# @app.route('/remove_edge', methods=['POST'])
# def remove_edge():
#     data = request.json
#     start_node = data['start_node']
#     end_node = data['end_node']
#     cg.remove_edge(start_node, end_node)
#     cg.visualize()  # 重新生成可视化文件
#     return jsonify({"status": "success", "message": f"Edge from {start_node} to {end_node} removed."})
#
#
# # 查询操作
# @app.route('/find_node', methods=['POST'])
# def find_node():
#     data = request.json
#     node_name = data['node_name']
#     node = cg.find_node(node_name)
#     return jsonify({"node": node})
#
#
# @app.route('/find_edge', methods=['POST'])
# def find_edge():
#     data = request.json
#     start_node = data['start_node']
#     end_node = data['end_node']
#     edge = cg.find_edge(start_node, end_node)
#     return jsonify({"edge": edge})
#
#
# @app.route('/visualize', methods=['GET'])
# def visualize():
#     # 不再生成文件，只返回成功状态
#     return jsonify({"status": "success"})


if __name__ == '__main__':



    # 设置日志记录


    app.run(debug=True, port=5002)
    # http://127.0.0.1:5001/