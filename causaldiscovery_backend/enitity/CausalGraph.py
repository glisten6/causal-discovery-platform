import os
import time
import logging
import networkx as nx
from pyvis.network import Network
from causalnex.structure import StructureModel
from networkx.algorithms import dag
from causaldiscovery_backend.enitity.Edge import Edge
from causaldiscovery_backend.enitity.Node import Node

class CausalGraph:
    def __init__(self,Nodes:list(Node),Edges:list(Edge)):
        self.sm = StructureModel()
        self.log = []
        # 创建初始因果图
        # 土壤肥力（SoilFertility）：影响作物产量。
        # 土壤酸碱度（SoilAcidity）：影响土壤肥力。
        # 气候条件（Climate）：影响作物产量。
        # 地形地势（Terrain）：影响土壤肥力。
        # 灌溉（Irrigation）：影响作物产量。
        # 病虫害（PestInfestation）：影响作物产量。
        # 作物产量（CropYield）：最终结果变量。

        for Node in Nodes:
            pass

        # self.add_node("SoilAcidity", type="EnvironmentalFactor", source="SoilTest")
        # self.add_node("SoilFertility", type="EnvironmentalFactor", source="SoilTest")
        # self.add_node("CropYield", type="Outcome", source="HarvestAnalysis")
        # self.add_node("Climate", type="EnvironmentalFactor", source="WeatherStation")
        # self.add_node("Terrain", type="EnvironmentalFactor", source="TopographicSurvey")
        # self.add_node("Irrigation", type="Intervention", source="FarmManagement")
        # self.add_node("PestInfestation", type="EnvironmentalFactor", source="PestSurvey")
        # 建立因果关系
        # 土壤肥力（SoilFertility）对作物产量（CropYield）有正向影响。
        # 土壤酸碱度（SoilAcidity）对土壤肥力（SoilFertility）有负向影响。
        # 气候条件（Climate）对作物产量（CropYield）有正向影响。
        # 地形地势（Terrain）对土壤肥力（SoilFertility）有负向影响。
        # 灌溉（Irrigation）对作物产量（CropYield）有正向影响。
        # 病虫害（PestInfestation）对作物产量（CropYield）有负向影响。
        self.add_edge("SoilAcidity", "SoilFertility", relation="Negative", source="AgriculturalResearch")
        self.add_edge("SoilFertility", "CropYield", relation="Positive", source="AgriculturalResearch")
        self.add_edge("Climate", "CropYield", relation="Positive", source="AgriculturalResearch")
        self.add_edge("Terrain", "SoilFertility", relation="Negative", source="AgriculturalResearch")
        self.add_edge("Irrigation", "CropYield", relation="Positive", source="AgriculturalResearch")
        self.add_edge("PestInfestation", "CropYield", relation="Negative", source="AgriculturalResearch")
        self.add_node("土壤肥力",attrs={})
    def add_node(self, node_name, **attrs):
        if node_name not in self.sm.nodes:
            self.sm.add_node(node_name)
            nx.set_node_attributes(self.sm, {node_name: attrs})
            logging.info(f"Added node: {node_name} with attributes {attrs}")
        else:
            logging.warning(f"Node {node_name} already exists.")

    def update_node(self, node_name, **attrs):
        if node_name in self.sm.nodes:
            current_attrs = self.sm.nodes[node_name]
            for key, value in attrs.items():
                current_attrs[key] = value
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

    def update_edge(self, start_node, end_node, **attrs):
        if self.sm.has_edge(start_node, end_node):
            current_attrs = self.sm.get_edge_data(start_node, end_node)
            for key, value in attrs.items():
                current_attrs[key] = value
            nx.set_edge_attributes(self.sm, {(start_node, end_node): current_attrs})
            logging.info(f"Updated edge from {start_node} to {end_node} with attributes {attrs}")
        else:
            logging.warning(f"Edge from {start_node} to {end_node} not found.")

    def remove_node(self, node_name):
        if node_name in self.sm.nodes:
            self.sm.remove_node(node_name)
            logging.info(f"Removed node: {node_name}")
        else:
            logging.warning(f"Node {node_name} not found.")

    def remove_edge(self, start_node, end_node):
        if self.sm.has_edge(start_node, end_node):
            self.sm.remove_edge(start_node, end_node)
            logging.info(f"Removed edge from {start_node} to {end_node}")
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

    def visualize(self):
        net = Network(notebook=False, height="600px", width="100%")
        for node in self.sm.nodes:
            net.add_node(node, label=node, title=str(self.sm.nodes[node]))
        for edge in self.sm.edges:
            relation_info = self.sm.get_edge_data(edge[0], edge[1])
            net.add_edge(edge[0], edge[1], label=relation_info.get('relation', 'Unknown'), arrows="to")
        net.save_graph("static/causal_graph.html")
        logging.info("Causal graph visualization saved to static/causal_graph.html")