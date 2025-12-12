from causaldiscovery_backend.enitity.Node import Node


class Edge:


    def __init__(self,sourceNode:Node,targetNode:Node,edgeId:int = -1,**kwargs):
        self.attrs = {}
        self.sourceNode = sourceNode
        self.targetNode = targetNode
        self.edgeId = edgeId
        self.attrs.update(**kwargs)




if __name__ == '__main__':
    node1 = Node('node1')
    node2 = Node('node2')
    edge = Edge(node1,node2,-1,relationship = 1)
    print(edge.attrs)
