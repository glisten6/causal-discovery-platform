import inspect
import io
import traceback
from tempfile import SpooledTemporaryFile
from tkinter import Image

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os 
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from causaldiscovery_backend.causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causaldiscovery_backend.causalnex.structure.pytorch import from_pandas


#from causaldiscovery_backend import

class Causal_Discovery:


    def __init__(self):
        self.sm = None

    def __stream_to_string__(self,file_stream, encoding='utf-8'):
        if isinstance(file_stream, io.BytesIO):
            # 二进制流 → 读取字节 → 解码为字符串
            file_stream.seek(0)  # 重置指针到起始位置
            byte_data = file_stream.read()
            return byte_data.decode(encoding)
        elif isinstance(file_stream, io.StringIO):
            # 文本流 → 直接读取字符串
            file_stream.seek(0)
            return file_stream.read()
        elif isinstance(file_stream, bytes):
            # 字节数据 → 直接解码
            return file_stream.decode(encoding)
        elif isinstance(file_stream, SpooledTemporaryFile):
            file_stream.seek(0)
            content = file_stream.read()
            return io.BytesIO(content)
        else:
            raise ValueError("Unsupported stream type")

    def get_target_column_index(self,df: pd.DataFrame, target=None) -> str:
        """
        获取 DataFrame 中目标列的索引

        参数:
            df (pd.DataFrame): 输入数据框
            target (str, int, None): 目标列名或索引，默认取最后一列

        返回:
            int: 目标列的索引

        异常:
            ValueError: 列名不存在或索引越界时抛出
            TypeError: target 类型不合法时抛出
        """
        # 默认情况：取最后一列
        if target is None:
            return [df.columns[-1]]

        # 列名类型（字符串）
        if isinstance(target, list):
            if not ( set(target) - set(df.columns)):
                return target
            else:
                raise ValueError(f"列名 '{target}' 不存在，可用列名为: {list(df.columns)}")

        # 索引类型（整数）
        elif isinstance(target, int):
            n_cols = df.shape[1]
            if 0 <= target < n_cols:
                return df.columns[target]
            else:
                raise IndexError(f"索引 {target} 越界，最大允许索引为 {n_cols - 1}")
        elif isinstance(target,str):
            if target  in df.columns:
                return [target]
            else:
                raise IndexError(f"{target} 不存在")

                # 非法类型
        else:
            raise TypeError("target 必须是字符串（列名）或整数（索引）")

    #  tabu_edges: list of edges(from, to) not to be included in the graph.

    #      tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.

    #    tabu_child_nodes: list of nodes banned from being a child of any other nodes.

    #   use_gpu: use gpu if it is set to True and CUDA is available.

    def discover_file(self,file,hasHead = True,cols=None,sep=None,target = None,use_model="pc"):
        def check_and_assign_header(df):
            # 检查是否没有表头（即列名是默认的整数索引）
            if isinstance(df.columns, pd.RangeIndex):
                # 为 DataFrame 赋予数字字符形式的表头
                df.columns = [str(i + 1) for i in range(len(df.columns))]
            return df
        df = None
        try:

            if hasHead:
                file = self.__stream_to_string__(file)
                df = pd.read_csv(file,sep=sep,engine='python')
            else:
                if cols is None:
                    raise ValueError("没有表头必须提供表头")

                df = pd.read_csv(file,sep=sep,names=cols,header=None,engine='python')
            df = check_and_assign_header(df)
            non_numeric_columns = list(df.select_dtypes(exclude=[np.number]).columns)
            le = LabelEncoder()
            for col in non_numeric_columns:
                df[col] = le.fit_transform(df[col])
            target = self.get_target_column_index(df,target)




            sm = from_pandas(df,tabu_parent_nodes= target,use_model=use_model)
            sm.remove_edges_below_threshold(0.8)
            viz = plot_structure(
                sm,
                all_node_attributes=NODE_STYLE.WEAK,
                all_edge_attributes=EDGE_STYLE.WEAK,
            )
            # 生成HTML内容
            html_content = viz.generate_html()
            self.sm = sm
            import os
            # 获取当前工作目录
            current_working_directory = os.getcwd()

            ##  手动写入文件
            with open("templates/network.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            return self.sm
        except Exception as e:
            print(traceback.format_exc())

    def analysis_features_use_current_structure(self):
        pass


