"""Unit tests for all algorithms and functionalities"""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph import DroneNetwork, Node, Edge
from src.algorithms import GraphAlgorithms


@pytest.fixture
def small_network():
    """Create small test network"""
    network = DroneNetwork()
    
    # Add nodes
    nodes = [
        Node("HUB", "Central Hub", "hub", 0, 0),
        Node("D1", "Delivery 1", "delivery", 5, 2),
        Node("D2", "Delivery 2", "delivery", 7, 4),
        Node("CH1", "Charging 1", "charging", 3, 1)
    ]
    for node in nodes:
        network.add_node(node)
    
    # Add edges
    edges = [
        Edge("HUB", "CH1", energy_cost=5, capacity=3, distance=3, bidirectional=True),
        Edge("CH1", "D1", energy_cost=6, capacity=2, distance=4),
        Edge("CH1", "D2", energy_cost=8, capacity=2, distance=5),
        Edge("D1", "D2", energy_cost=4, capacity=1, distance=2, bidirectional=True)
    ]
    for edge in edges:
        network.add_edge(edge)
    
    return network


def test_load_save_json(tmp_path):
    """Test B1: Import/Export JSON"""
    network = DroneNetwork()
    network.add_node(Node("A", "Node A", "hub"))
    network.add_edge(Edge("A", "A", energy_cost=10, capacity=5, distance=10))
    
    file_path = tmp_path / "test.json"
    network.save_to_json(str(file_path))
    
    network2 = DroneNetwork()
    network2.load_from_json(str(file_path))
    
    assert len(network2.nodes) == 1
    assert len(network2.edges) == 1


def test_no_fly_zones(small_network):
    """Test B2: No-fly zones"""
    # Mark as restricted
    assert small_network.mark_restricted("HUB", "CH1") == True
    assert small_network.is_restricted("HUB", "CH1") == True
    
    # Unmark
    assert small_network.unmark_restricted("HUB", "CH1") == True
    assert small_network.is_restricted("HUB", "CH1") == False


def test_modify_network(small_network):
    """Test B3: Modify network"""
    # Add node
    new_node = Node("D3", "Delivery 3", "delivery", 10, 10)
    small_network.add_node(new_node)
    assert "D3" in small_network.nodes
    
    # Add edge
    new_edge = Edge("D2", "D3", energy_cost=15, capacity=2, distance=8)
    small_network.add_edge(new_edge)
    assert len(small_network.edges) == 5
    
    # Update energy cost
    assert small_network.update_energy_cost("HUB", "CH1", 10.0) == True


def test_dijkstra_shortest_path(small_network):
    """Test F2: Dijkstra's algorithm"""
    algo = GraphAlgorithms()
    distance, path = algo.dijkstra(small_network, "HUB", "D1", use_energy=False)
    
    assert path is not None
    assert path[0] == "HUB"
    assert path[-1] == "D1"
    assert distance == 7  # HUB -> CH1 (3) -> D1 (4)


def test_dijkstra_energy_optimization(small_network):
    """Test F2: Energy-based routing"""
    algo = GraphAlgorithms()
    energy, path = algo.dijkstra(small_network, "HUB", "D2", use_energy=True)
    
    assert path is not None
    assert energy == 13  # HUB -> CH1 (5) -> D2 (8)


def test_bfs_reachability(small_network):
    """Test F1: BFS reachability"""
    algo = GraphAlgorithms()
    
    assert algo.bfs_reachability(small_network, "HUB", "D2") == True
    assert algo.bfs_reachability(small_network, "D1", "HUB") == True  # bidirectional


def test_kruskal_mst(small_network):
    """Test F6: Kruskal's MST"""
    algo = GraphAlgorithms()
    mst_edges, total_cost = algo.kruskal_mst(small_network, use_energy=False)
    
    # MST should have n-1 edges
    assert len(mst_edges) == 3
    assert total_cost > 0


def test_ford_fulkerson(small_network):
    """Test F3: Ford-Fulkerson max flow"""
    algo = GraphAlgorithms()
    max_flow = algo.ford_fulkerson(small_network, "HUB", "D1")
    
    assert max_flow == 2  # Limited by CH1->D1 capacity


def test_connectivity_analysis(small_network):
    """Test F4: Connectivity analysis"""
    algo = GraphAlgorithms()
    analysis = algo.analyze_connectivity(small_network)
    
    assert 'average_connectivity' in analysis
    assert 'critical_nodes' in analysis
    assert len(analysis['connectivity_map']) == 4


def test_bridge_finding(small_network):
    """Test F4: Find bridges"""
    algo = GraphAlgorithms()
    bridges = algo.find_bridges(small_network)
    
    # Bridges are edges whose removal disconnects graph
    assert isinstance(bridges, list)


def test_charging_coverage(small_network):
    """Test F5: Charging coverage analysis"""
    algo = GraphAlgorithms()
    analysis = algo.analyze_charging_coverage(small_network)
    
    assert 'average_distance' in analysis
    assert 'max_distance' in analysis
    assert 'coverage_quality' in analysis


def test_restricted_edges_avoid(small_network):
    """Test that algorithms avoid restricted edges"""
    algo = GraphAlgorithms()
    
    # Mark edge as restricted
    small_network.mark_restricted("CH1", "D1")
    
    # Dijkstra should find alternative path or fail
    distance, path = algo.dijkstra(small_network, "HUB", "D1", avoid_restricted=True)
    
    # Should either find alternative or return None
    if path:
        # Path should not use restricted edge
        assert not ("CH1" in path and path[path.index("CH1") + 1] == "D1")


def test_bidirectional_edges(small_network):
    """Test bidirectional edge handling"""
    algo = GraphAlgorithms()
    
    # Should be able to go both ways on bidirectional edge
    distance1, path1 = algo.dijkstra(small_network, "HUB", "CH1")
    distance2, path2 = algo.dijkstra(small_network, "CH1", "HUB")
    
    assert path1 is not None
    assert path2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])