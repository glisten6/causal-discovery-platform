import json
from pyecharts.charts import Tree
from pyecharts import options as opts
import os  
from datetime import datetime

class VersionNode:
    def __init__(self, version_number, content='',  author='Unknown', parent=None, causal_graph_hash=None):
        self.version_number = version_number
        self.content = content
        self.parent = parent
        self.children = []
        self.author = author
        self.created_at = datetime.now()  # 创建时间
        self.last_modified = datetime.now()  # 最近修改时间
        self.causal_graph_hash = causal_graph_hash  # 新增: 存储因果图文件哈希值
        # if not self.causal_graph_hash:
        #     self.causal_graph_hash = self.generate_causal_graph_hash()

    def add_child(self, child_node):
        self.children.append(child_node)

    def update_content(self, new_content):
        """更新节点内容并更新最后修改时间"""
        self.content = new_content
        self.last_modified = datetime.now()
        return self

    def to_dict(self):
        """递归将节点转换为字典"""
        return {
            'version_number': self.version_number,
            'content': self.content,
            'author': self.author,
            'created_at': self.created_at.isoformat(),  # 保存为ISO格式的字符串
            'last_modified': self.last_modified.isoformat(),  # 保存为ISO格式的字符串
            'causal_graph_hash': self.causal_graph_hash,  # 添加哈希值字段
            'children': [child.to_dict() for child in self.children]
        }

    @staticmethod
    def from_dict(data, parent=None):
        """从字典递归重建节点（增加时间信息）"""
        node = VersionNode(
            version_number=data['version_number'],
            content=data['content'],
            author=data.get('author', 'Unknown'),
            parent=parent,
            causal_graph_hash=data.get('causal_graph_hash')  # 添加哈希值字段
        )
        
        # 设置创建时间和修改时间
        if 'created_at' in data:
            try:
                node.created_at = datetime.fromisoformat(data['created_at'])
            except:
                node.created_at = datetime.now()
        
        if 'last_modified' in data:
            try:
                node.last_modified = datetime.fromisoformat(data['last_modified'])
            except:
                node.last_modified = datetime.now()
        
        for child_data in data.get('children', []):
            child_node = VersionNode.from_dict(child_data, parent=node)
            node.add_child(child_node)
        return node

class VersionTree:
    def __init__(self, initial_content='Initial Version', variety_name = '', path = 'Unknown', author='Unknown'):
        self.root = VersionNode('v1.0', initial_content, author= author)
        self.current = self.root
        self.version_counter = 1  # 版本计数器，从1开始
        self.variety_name =  variety_name  # 种类名称，也作为版本树的名称
        # self.path = path          # 保存目录， 加上variety_name为保存路径

    def _generate_next_version_number(self):
        """生成下一个版本号"""
        self.version_counter += 1
        return f"v{self.version_counter}.0"

    def create_version(self, content='', version_number=None, author='Unknown', causal_graph_hash=None):
        """创建新版本节点，自动生成版本号"""
        version_number = self._generate_next_version_number()

        new_node = VersionNode(version_number, content, author=author, parent=self.current, causal_graph_hash=causal_graph_hash)
        self.current.add_child(new_node)
        self.current = new_node
        return new_node
    
    def update_node_content(self, version_number, new_content):
        """更新指定节点的内容并记录修改时间"""
        node = self.find_version(self.root, version_number)
        if node is None:
            raise ValueError(f"版本 {version_number} 不存在")
        
        return node.update_content(new_content)


    def create_branch(self, from_version_number, content='', author='Unknown'):
        """从指定版本创建分支，自动生成版本号"""
        from_node = self.find_version(self.root, from_version_number)
        if from_node is None:
            raise ValueError(f"版本 {from_version_number} 不存在")

        # 分支也使用顺序版本号
        version_number = self._generate_next_version_number()
        new_node = VersionNode(version_number, content, author=author, parent=from_node)
        from_node.add_child(new_node)
        self.current = new_node
        return new_node

    def rollback(self, version_number):
        """回滚到指定版本"""
        node = self.find_version(self.root, version_number)
        if node is None:
            raise ValueError(f"版本 {version_number} 不存在")

        self.current = node
        return node
    
    def find_version_by_version_number(self, node, version_number):
        """递归查找指定版本号的节点
        
        Args:
            node: 开始搜索的节点
            version_number: 要查找的版本号，如 'v1.0'
            
        Returns:
            找到的节点或None
        """
        version_number = version_number.replace('(当前)', '').strip()   # 去掉"(当前)"标记和空格

        # 根据版本号查找
        if node.version_number == version_number:
            return node
        
        # 递归查找子节点
        for child in node.children:
            found = self.find_version_by_version_number(child, version_number)
            if found:
                return found
        return None

    def find_version(self, node, version_identifier):
        """递归查找节点（支持使用版本号或哈希值查找）
        
        Args:
            node: 开始搜索的节点
            version_identifier: 版本标识符(版本号或哈希值)
            
        Returns:
            找到的节点或None
        """
        # 如果是版本号格式，使用版本号查找
        if isinstance(version_identifier, str) and version_identifier.startswith('v'):
            return self.find_version_by_version_number(node, version_identifier)
        
        # 否则认为是哈希值，根据哈希值查找
        if node.causal_graph_hash and node.causal_graph_hash == version_identifier:
            return node
        
        # 递归查找子节点
        for child in node.children:
            found = self.find_version(child, version_identifier)
            if found:
                return found
        return None    

    def get_hash_by_version(self, version_number):
        """根据版本号查找对应节点的哈希值
        
        Args:
            version_number: 版本号，如'v1.0'
            
        Returns:
            对应节点的哈希值，如果节点不存在则抛出ValueError，如果没有哈希值则返回None
        """
        node = self.find_version_by_version_number(self.root, version_number)
        if node is None:
            raise ValueError(f"版本 {version_number} 不存在")
        
        return node.causal_graph_hash

    # display方法增加显示作者
    def display(self, node=None, depth=0):
        if node is None:
            node = self.root

        prefix = '    ' * depth + ('└─' if depth > 0 else '')
        current_marker = ' (current)' if node == self.current else ''
        modified_time = node.last_modified.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{prefix}{node.version_number}{current_marker}: {node.content} [作者: {node.author}] [最后修改: {modified_time}]")

        for child in node.children:
            self.display(child, depth + 1)

    def save_to_file(self, filename):
        """保存树结构到JSON文件"""
        # if not self.path:
        #     print("警告：路径不存在，将使用默认路径保存文件")
        #     # 可以设置一个默认路径或者直接使用当前目录
        #     self.path = "."
        
        if not self.variety_name:
            print("警告：种类名称不存在，建议设置种类名称以便于管理, 当前无法保存")
            # 可以选择设置一个默认值或者仅提示
            # self.variety_name = "未命名种类"
            return 
        
        tree_data = {
            'root': self.root.to_dict(),
            'current_version': self.current.version_number,
            'version_counter': self.version_counter,
            'variety_name': self.variety_name,  # 保存种类信息
            # 'path': self.path  # 同时保存路径信息
        }
        
        # 确保路径存在
        os.makedirs(os.path.dirname(os.path.join(filename)), exist_ok=True)
        
        # 使用完整路径保存文件
        full_path = filename
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, indent=4, ensure_ascii=False)
        
        print(f"文件已保存至: {full_path}")


    @staticmethod
    def load_from_file(filename):
        """从JSON文件加载树结构"""
        with open(filename, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)

        tree = VersionTree()
        tree.root = VersionNode.from_dict(tree_data['root'])
        tree.current = tree.find_version(tree.root, tree_data['current_version'])
        tree.version_counter = tree_data['version_counter']
        tree.variety_name = tree_data.get('variety_name', '')  # 加载种类信息
        return tree

    def build_pyecharts_tree_data(self, node):
        """
        将自定义的树节点转换为 pyecharts Tree 所需的数据结构（递归）。
        返回值是一个 dict，每个子节点使用 children 列表表示。
        """
        modified_time = node.last_modified.strftime("%Y-%m-%d %H:%M:%S")
        
        # 检查是否为当前节点
        is_current = node == self.current
        
        result = {
            "name": f"{node.version_number}",
            "tooltip": f"版本号: {node.version_number}<br>内容: {node.content}<br>作者: {node.author}<br>最后修改: {modified_time}",
            "children": [self.build_pyecharts_tree_data(child) for child in node.children]
        }
        
        # 为当前节点添加标记
        if is_current:
            result["itemStyle"] = {"color": "#F56C6C"}  # 设置为红色
            result["name"] = f"{node.version_number}"  # 在名称中标注"当前"

        # TODO: 为最新节点添加标记
        if f'v{self.version_counter}.0' in node.version_number:
            result["name"] = f"{node.version_number} (最新)" 
        
        return result

    def visualize_version_control_tree(self, output_file="version_control_tree.html"):
        """
        使用 pyecharts 可视化指定的版本管理树，并保存为 HTML 文件。
        """
        if not self.root:
            raise ValueError("版本管理树中没有根节点，无法可视化！")

        # 构建 pyecharts Tree 数据
        data = [self.build_pyecharts_tree_data(self.root)]  # 输入根节点

        c = (
            Tree()
            .add(
                series_name="版本管理树",
                data=data,
                orient="LR",  # 可选："TB" (自上而下), "LR" (自左而右), "RL" 等
                symbol="circle",
                symbol_size=36,  # 放大节点
                is_expand_and_collapse=False,
                label_opts=opts.LabelOpts(
                    position="top",
                    vertical_align="middle",
                    font_size=20  # 放大字体
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="item",
                    formatter="{b}<br/>{c}"
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="版本管理树",
                    subtitle=f"种类: {self.variety_name}" if self.variety_name else "",
                    title_textstyle_opts=opts.TextStyleOpts(font_size=24),  # 放大标题
                    subtitle_textstyle_opts=opts.TextStyleOpts(font_size=18)  # 放大副标题
                ),
                legend_opts=opts.LegendOpts(
                    pos_left="left",
                    orient="vertical"
                )
            )
        )
        
        # 添加图例说明
        c.set_series_opts(
            label_opts=opts.LabelOpts(formatter="{b}")
        )
        
        c.render(output_file)
        print(f"可视化完成！HTML 文件已保存为：{output_file}")

    @classmethod
    def parse_html_to_tree(cls, html_file):
        """
        从HTML文件解析回版本管理树结构
        
        Args:
            html_file: HTML文件路径
        
        Returns:
            版本树实例
        """
        from bs4 import BeautifulSoup
        import json
        import re
        
        # 读取HTML文件
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 查找JSON数据 - pyecharts通常将数据存储在JavaScript代码中
        script_tags = soup.find_all('script')
        tree_data = None
        
        # 调试: 打印找到的script数量
        print(f"找到{len(script_tags)}个script标签")
        
        for i, script in enumerate(script_tags):
            if script.string and 'option' in script.string:
                print(f"在第{i+1}个script标签中找到option")
                try:
                    # 提取data部分 - 使用更宽松的正则表达式
                    # 首先尝试提取option对象
                    option_match = re.search(r'var\s+option\s*=\s*({.*?});', script.string, re.DOTALL)
                    if not option_match:
                        # 尝试其他可能的模式
                        option_match = re.search(r'option\s*=\s*({.*?});', script.string, re.DOTALL)
                    
                    if option_match:
                        option_str = option_match.group(1)
                        print("成功提取option对象")
                        
                        # 在option中查找data数组
                        data_match = re.search(r'"data"\s*:\s*(\[.*?\])', option_str, re.DOTALL)
                        if data_match:
                            data_str = data_match.group(1)
                            print("成功提取data数组")
                            
                            # 直接使用eval处理JavaScript对象
                            # 注意：在生产环境中，eval可能存在安全风险
                            import ast
                            # 清理字符串，将JS对象转为Python可解析的格式
                            data_str = data_str.replace('true', 'True').replace('false', 'False')
                            data_str = re.sub(r'(\w+):', r'"\1":', data_str)  # 为键名添加引号
                            data_str = data_str.replace("'", '"')  # 统一使用双引号
                            
                            try:
                                # 尝试使用ast.literal_eval更安全地解析
                                tree_data = ast.literal_eval(data_str)
                                print("成功解析data数组")
                                break
                            except:
                                # 如果ast.literal_eval失败，尝试使用json.loads
                                try:
                                    tree_data = json.loads(data_str)
                                    print("使用json.loads成功解析data数组")
                                    break
                                except:
                                    print("无法解析data数组")
                                    continue
                except Exception as e:
                    print(f"解析脚本时出错: {e}")
                    continue
        
        if not tree_data:
            # 尝试一种更简单的方法 - 直接从文件名找到对应的JSON文件
            json_file = html_file.replace('.html', '.json')
            if os.path.exists(json_file):
                print(f"从HTML无法提取数据，尝试从对应的JSON文件加载: {json_file}")
                return cls.load_from_file(json_file)
            raise ValueError("无法从HTML中提取树数据")
        
        # 创建新的版本树实例
        parsed_tree = cls('由HTML解析')
        
        # 递归构建版本树节点
        def extract_version_info(name_str):
            """从节点名称中提取版本号、内容和作者"""
            version = name_str
            content = ""
            author = "Unknown"
            
            # 尝试从tooltip中提取更多信息
            tooltip = node_data.get("tooltip", "")
            if tooltip:
                # 从tooltip中提取内容和作者
                content_match = re.search(r'内容:\s*(.*?)<br>', tooltip)
                if content_match:
                    content = content_match.group(1)
                
                author_match = re.search(r'作者:\s*(.*)', tooltip)
                if author_match:
                    author = author_match.group(1)
            
            return version, content, author
        
        def build_tree_from_data(node_data, parent=None):
            """递归构建树结构"""
            name = node_data.get("name", "")
            version, content, author = extract_version_info(name)
            
            if parent is None:
                # 根节点
                parsed_tree.root.version_number = version
                parsed_tree.root.content = content
                parsed_tree.root.author = author
                node = parsed_tree.root
            else:
                # 创建子节点
                # 这里简化处理，直接创建VersionNode而不使用create_version/create_branch方法
                node = VersionNode(version, content, author=author, parent=parent)
                parent.add_child(node)
            
            # 处理子节点
            children = node_data.get("children", [])
            for child_data in children:
                build_tree_from_data(child_data, node)
            
            return node
        
        # 处理根节点
        if tree_data and len(tree_data) > 0:
            root_node = build_tree_from_data(tree_data[0])
            parsed_tree.current = parsed_tree.root  # 设置当前节点为根节点
            
            # 设置版本计数器
            max_version = 1
            def update_version_counter(node):
                nonlocal max_version
                if node.version_number.startswith('v'):
                    try:
                        ver_num = float(node.version_number[1:].split('-')[0])
                        max_version = max(max_version, int(ver_num))
                    except:
                        pass
                for child in node.children:
                    update_version_counter(child)
            
            update_version_counter(parsed_tree.root)
            parsed_tree.version_counter = max_version
            
            return parsed_tree
        
        raise ValueError("无法从HTML中构建树结构")




if __name__ == "__main__":
    # 创建版本树实例，添加种类名称
    vtree = VersionTree('初始版本', variety_name='水稻品种A', path='./data')

    # 创建版本并指定作者 - 不再传入版本号参数
    v2_node = vtree.create_version('第二个版本内容更新', author='张三')
    print(f"创建的第二个版本号为: {v2_node.version_number}")  # 应该显示v2.0
    
    v3_node = vtree.create_version('第三个版本内容更新', author='李四')
    print(f"创建的第三个版本号为: {v3_node.version_number}")  # 应该显示v3.0

    # 测试更新节点内容功能 - 这将更新修改时间
    print("\n更新节点内容测试：")
    print(f"更新前，v2节点修改时间: {v2_node.last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
    # 等待一段时间以便清楚地看到时间变化
    import time
    time.sleep(2)
    # 更新内容 - 使用v2_node.version_number而不是硬编码的'v2.0'
    vtree.update_node_content(v2_node.version_number, '更新后的第二个版本内容')
    print(f"更新后，v2节点修改时间: {v2_node.last_modified.strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建分支版本并指定作者 - 不再传入分支版本号
    branch_node = vtree.create_branch('v1.0', '第一个版本的分支版本', author='王五')
    print(f"创建的分支版本号为: {branch_node.version_number}")  # 应该显示v4.0

    # 测试回滚功能
    vtree.rollback('v1.0')
    print(f"回滚到版本: {vtree.current.version_number}")
    
    # 从回滚点创建新版本
    new_branch = vtree.create_version('从v1.0创建的新版本', author='赵六')
    print(f"从回滚点创建的新版本号为: {new_branch.version_number}")  # 应该显示v5.0

    # 显示当前树结构，包含最后修改时间
    print("\n当前版本树结构（包含种类和修改时间）：")
    print(f"种类名称: {vtree.variety_name}")
    vtree.display()

    # 保存到文件，这会检查种类名称
    print("\n保存文件测试：")
    vtree.save_to_file('水稻品种A_version_tree.json')

    # 测试无种类名称的情况
    print("\n测试无种类名称的保存：")
    no_variety_tree = VersionTree('无种类的初始版本')
    no_variety_tree.save_to_file('no_variety_tree.json')  # 这应该会显示警告并不保存

    # 从文件读取版本树
    print("\n从文件加载测试：")
    loaded_vtree = VersionTree.load_from_file('./data/水稻品种A_version_tree.json')
    print(f"加载的版本树种类名称: {loaded_vtree.variety_name}")
    print(f"加载的版本树当前节点: {loaded_vtree.current.version_number}")
    print(f"加载的版本树当前版本计数器: {loaded_vtree.version_counter}")
    loaded_vtree.display()

    # 在加载的树上继续创建新版本，测试版本计数器是否正确延续
    new_loaded_version = loaded_vtree.create_version('加载后创建的新版本', author='钱七')
    print(f"加载后创建的新版本号为: {new_loaded_version.version_number}")  # 应该是v6.0
    
    # 可视化版本树，保存html
    print("\n可视化测试：")
    vtree.visualize_version_control_tree(output_file="./data/水稻品种A_version_tree.html")

    # 测试从HTML文件解析回树结构
    print("\n从HTML解析回树结构测试：")
    try:
        html_tree = VersionTree.parse_html_to_tree("./data/水稻品种A_version_tree.html")
        print("从HTML解析成功！")
        print("解析后的树结构：")
        html_tree.display()
        
        # 验证解析的树结构与原始结构是否一致
        print("\n验证原始树与解析后的树是否一致：")
        
        def compare_trees(original_node, parsed_node, path="根"):
            """比较两个树节点及其子树是否一致"""
            # 使用version_number而不是name
            if original_node.version_number != parsed_node.version_number:
                print(f"版本号不一致 在 {path}: {original_node.version_number} vs {parsed_node.version_number}")
                return False
            
            # 检查内容是否一致
            if original_node.content != parsed_node.content:
                print(f"内容不一致 在 {path}: {original_node.content} vs {parsed_node.content}")
                return False
            
            # 检查作者是否一致
            if original_node.author != parsed_node.author:
                print(f"作者不一致 在 {path}: {original_node.author} vs {parsed_node.author}")
                return False
            
            # 检查子节点数量
            if len(original_node.children) != len(parsed_node.children):
                print(f"子节点数量不一致 在 {path}: {len(original_node.children)} vs {len(parsed_node.children)}")
                return False
            
            # 递归比较所有子节点
            result = True
            for i, (original_child, parsed_child) in enumerate(zip(original_node.children, parsed_node.children)):
                child_path = f"{path} > {original_child.version_number}"
                if not compare_trees(original_child, parsed_child, child_path):
                    result = False
            
            return result
        
        if compare_trees(vtree.root, html_tree.root):
            print("验证成功！原始树与从HTML解析的树结构一致。")
        else:
            print("验证失败！原始树与从HTML解析的树结构不一致。")
            
    except Exception as e:
        print(f"从HTML解析回树结构失败: {e}")