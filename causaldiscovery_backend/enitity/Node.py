from causaldiscovery_backend.enitity import Node
# 因果发现的节点
# 包含节点名、
class Node:


    def __init__(self, nodeName, nodeId = -1, **kwargs):
        self.attrs = {}
        self.nodeID = nodeId
        self.nodeName = nodeName
        self.attrs.update(**kwargs)






if __name__ == '__main__':

    node = Node(nodeId=0,nodeName="土壤",type="EnvironmentalFactor", source="SoilTest")














