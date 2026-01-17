"""Graph data structure - Implements B1 and B3"""
import json
from typing import List, Dict, Optional, Tuple, Set


class Node:
    """Represents a node in the drone network"""
    
    def __init__(self, node_id: str, name: str, node_type: str, x: float = 0, y: float = 0):
        self.id = node_id
        self.name = name
        self.type = node_type  # hub, delivery, charging, relay
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Node({self.id}, {self.type})"
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


class Edge:
    """Represents a flight corridor"""
    
    def __init__(self, from_node: str, to_node: str, 
                 energy_cost: float = 0, capacity: int = 0, 
                 distance: float = 0, bidirectional: bool = False,
                 restricted: bool = False):
        self.from_node = from_node
        self.to_node = to_node
        self.energy_cost = energy_cost
        self.capacity = capacity
        self.distance = distance
        self.bidirectional = bidirectional
        self.restricted = restricted  # B2: No-fly zone support
    
    def __repr__(self):
        direction = "<->" if self.bidirectional else "->"
        status = " (RESTRICTED)" if self.restricted else ""
        return f"Edge({self.from_node} {direction} {self.to_node}{status})"


class DroneNetwork:
    """
    Complete graph representation
    
    DATA STRUCTURES USED:
    - Dictionary (Hash Map): O(1) node lookup
    - List: Store all edges
    - Adjacency List (Dict of Lists): O(1) neighbor lookup
    - Set: Track restricted edges
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}  # Hash Map for nodes
        self.edges: List[Edge] = []
        self.adjacency_list: Dict[str, List[Tuple[str, Edge]]] = {}  # Hash Map for adjacency
        self.restricted_edges: Set[Tuple[str, str]] = set()  # Hash Set for restrictions
    
    # ==================== B1: IMPORT/EXPORT ====================
    
    def load_from_json(self, filepath: str) -> None:
        """B1: Load network from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load nodes
        for node_data in data['nodes']:
            node = Node(
                node_data['id'],
                node_data.get('name', node_data['id']),
                node_data['type'],
                node_data.get('x', 0),
                node_data.get('y', 0)
            )
            self.add_node(node)
        
        # Load edges
        for edge_data in data['edges']:
            edge = Edge(
                edge_data['from'],
                edge_data['to'],
                edge_data.get('energy_cost', 0),
                edge_data.get('capacity', 0),
                edge_data.get('distance', 0),
                edge_data.get('bidirectional', False),
                edge_data.get('restricted', False)
            )
            self.add_edge(edge)
    
    def save_to_json(self, filepath: str) -> None:
        """B1: Save network to JSON file"""
        data = {
            'nodes': [
                {
                    'id': node.id,
                    'name': node.name,
                    'type': node.type,
                    'x': node.x,
                    'y': node.y
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'from': edge.from_node,
                    'to': edge.to_node,
                    'energy_cost': edge.energy_cost,
                    'capacity': edge.capacity,
                    'distance': edge.distance,
                    'bidirectional': edge.bidirectional,
                    'restricted': edge.restricted
                }
                for edge in self.edges
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    # ==================== B3: MODIFY NETWORK ====================
    
    def add_node(self, node: Node) -> None:
        """B3: Add a new node (charging station, delivery point, etc.)"""
        self.nodes[node.id] = node
        if node.id not in self.adjacency_list:
            self.adjacency_list[node.id] = []
    
    def add_edge(self, edge: Edge) -> None:
        """B3: Add a new flight corridor"""
        self.edges.append(edge)
        
        # Forward direction
        if edge.from_node not in self.adjacency_list:
            self.adjacency_list[edge.from_node] = []
        self.adjacency_list[edge.from_node].append((edge.to_node, edge))
        
        # Track restricted edges (B2)
        if edge.restricted:
            self.restricted_edges.add((edge.from_node, edge.to_node))
        
        # Bidirectional handling
        if edge.bidirectional:
            if edge.to_node not in self.adjacency_list:
                self.adjacency_list[edge.to_node] = []
            self.adjacency_list[edge.to_node].append((edge.from_node, edge))
            if edge.restricted:
                self.restricted_edges.add((edge.to_node, edge.from_node))
    
    def remove_edge(self, from_node: str, to_node: str) -> bool:
        """B3: Remove a flight corridor"""
        # Remove from edge list
        self.edges = [e for e in self.edges 
                     if not (e.from_node == from_node and e.to_node == to_node)]
        
        # Remove from adjacency list
        if from_node in self.adjacency_list:
            self.adjacency_list[from_node] = [
                (n, e) for n, e in self.adjacency_list[from_node] if n != to_node
            ]
        
        # Remove from restricted set
        self.restricted_edges.discard((from_node, to_node))
        return True
    
    def update_energy_cost(self, from_node: str, to_node: str, new_cost: float) -> bool:
        """B3: Update energy cost of a corridor"""
        for edge in self.edges:
            if edge.from_node == from_node and edge.to_node == to_node:
                edge.energy_cost = new_cost
                return True
        return False
    
    # ==================== B2: NO-FLY ZONES ====================
    
    def mark_restricted(self, from_node: str, to_node: str) -> bool:
        """B2: Mark a corridor as restricted (no-fly zone)"""
        for edge in self.edges:
            if edge.from_node == from_node and edge.to_node == to_node:
                edge.restricted = True
                self.restricted_edges.add((from_node, to_node))
                return True
        return False
    
    def unmark_restricted(self, from_node: str, to_node: str) -> bool:
        """B2: Remove no-fly zone restriction"""
        for edge in self.edges:
            if edge.from_node == from_node and edge.to_node == to_node:
                edge.restricted = False
                self.restricted_edges.discard((from_node, to_node))
                return True
        return False
    
    # ==================== HELPER METHODS ====================
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID - O(1) lookup using hash map"""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str, include_restricted: bool = True) -> List[Tuple[str, Edge]]:
        """Get neighbors - O(1) lookup using adjacency list"""
        neighbors = self.adjacency_list.get(node_id, [])
        if not include_restricted:
            neighbors = [(n, e) for n, e in neighbors if not e.restricted]
        return neighbors
    
    def is_restricted(self, from_node: str, to_node: str) -> bool:
        """Check if edge is restricted - O(1) using hash set"""
        return (from_node, to_node) in self.restricted_edges
    
    def get_statistics(self) -> Dict:
        """Get network statistics"""
        node_types = {}
        for node in self.nodes.values():
            node_types[node.type] = node_types.get(node.type, 0) + 1
        
        restricted_count = len([e for e in self.edges if e.restricted])
        bidirectional_count = len([e for e in self.edges if e.bidirectional])
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': node_types,
            'restricted_edges': restricted_count,
            'bidirectional_edges': bidirectional_count
        }
    
    def __str__(self):
        stats = self.get_statistics()
        return f"DroneNetwork(nodes={stats['total_nodes']}, edges={stats['total_edges']})"