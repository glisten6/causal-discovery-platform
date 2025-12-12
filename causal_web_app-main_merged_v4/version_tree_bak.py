import json
from pyecharts.charts import Tree
from pyecharts import options as opts
import os  

class VersionNode:
    def __init__(self, version_number, content='',  author='Unknown', parent=None):
        self.version_number = version_number
        self.content = content
        self.parent = parent
        self.children = []
        self.author = author

    def add_child(self, child_node):
        self.children.append(child_node)

    def to_dict(self):
        """递归将节点转换为字典"""
        return {
            'version_number': self.version_number,
            'content': self.content,
            'author': self.author,
            'children': [child.to_dict() for child in self.children]
        }

    @staticmethod
    def from_dict(data, parent=None):
        """从字典递归重建节点（增加author）"""
        node = VersionNode(
            version_number=data['version_number'],
            content=data['content'],
            author=data.get('author', 'Unknown'),  # 反序列化author
            parent=parent
        )
        for child_data in data.get('children', []):
            child_node = VersionNode.from_dict(child_data, parent=node)
            node.add_child(child_node)
        return node

class VersionTree:
    def __init__(self, initial_content='Initial Version'):
        self.root = VersionNode('v1.0', initial_content)
        self.current = self.root
        self.version_counter = 1


    def create_version(self, content='', version_number=None, author='Unknown'):
        if version_number is None:
            self.version_counter += 1
            version_number = f'v{self.version_counter}.0'

        new_node = VersionNode(version_number, content, author=author, parent=self.current)
        self.current.add_child(new_node)
        self.current = new_node
        return new_node


    def create_branch(self, from_version_number, content='', branch_version_number=None, author='Unknown'):
        from_node = self.find_version(self.root, from_version_number)
        if from_node is None:
            raise ValueError(f"版本 {from_version_number} 不存在")

        if branch_version_number is None:
            branch_version_number = f'{from_version_number}-branch'

        new_node = VersionNode(branch_version_number, content, author=author, parent=from_node)
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

    def find_version(self, node, version_number):
        """递归查找版本节点"""
        if node.version_number == version_number:
            return node
        for child in node.children:
            found = self.find_version(child, version_number)
            if found:
                return found
        return None


    # display方法增加显示作者
    def display(self, node=None, depth=0):
        if node is None:
            node = self.root

        prefix = '    ' * depth + ('└─' if depth > 0 else '')
        current_marker = ' (current)' if node == self.current else ''
        print(f"{prefix}{node.version_number}{current_marker}: {node.content} [作者: {node.author}]")

        for child in node.children:
            self.display(child, depth + 1)

    def save_to_file(self, filename):
        """保存树结构到JSON文件"""
        tree_data = {
            'root': self.root.to_dict(),
            'current_version': self.current.version_number,
            'version_counter': self.version_counter
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, indent=4, ensure_ascii=False)


    @staticmethod
    def load_from_file(filename):
        """从JSON文件加载树结构"""
        with open(filename, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)

        tree = VersionTree()
        tree.root = VersionNode.from_dict(tree_data['root'])
        tree.current = tree.find_version(tree.root, tree_data['current_version'])
        tree.version_counter = tree_data['version_counter']
        return tree

    def build_pyecharts_tree_data(self, node):
        """
        将自定义的树节点转换为 pyecharts Tree 所需的数据结构（递归）。
        返回值是一个 dict，每个子节点使用 children 列表表示。
        """
        return {
            "name": f"{node.version_number}",
            "tooltip": f"版本号: {node.version_number}<br>内容: {node.content}<br>作者: {node.author}",
            "children": [self.build_pyecharts_tree_data(child) for child in node.children]
        }

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
                symbol_size=14,
                is_expand_and_collapse=False,
                label_opts=opts.LabelOpts(
                    position="top",
                    vertical_align="middle",
                    font_size=12
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="item",
                    formatter="{b}<br/>{c}"
                )
            )
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
    # 创建版本树实例
    vtree = VersionTree('初始版本')

    # 创建版本并指定作者
    vtree.create_version('第二个版本内容更新', author='张三')
    vtree.create_version('第三个版本内容更新', author='李四')

    # 创建分支版本并指定作者
    vtree.create_branch('v1.0', '第一个版本的分支版本', author='王五')

    # 保存到文件
    vtree.save_to_file('version_tree.json')

    # 可视化版本树，保存html
    vtree.visualize_version_control_tree(output_file="version_tree.html")

    # 从文件读取版本树
    loaded_vtree = VersionTree.load_from_file('version_tree.json')

    # 显示树结构（现在包含作者信息）
    loaded_vtree.display()

    # 测试从HTML文件解析回树结构
    print("\n从HTML解析回树结构测试：")
    print("\n从HTML解析回树结构测试：")
    try:
        # 假设parse_html_to_tree已经实现为VersionTree的类方法
        html_tree = VersionTree.parse_html_to_tree("version_tree.html")
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
