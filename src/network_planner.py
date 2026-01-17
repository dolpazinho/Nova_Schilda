"""High-level interface implementing all functionalities"""
import math
from src.graph import DroneNetwork, Node, Edge
from src.algorithms import GraphAlgorithms


class NetworkPlanner:
    """
    Main planning interface for drone network
    Implements all B1-B3 and F1-F6 functionalities
    """
    
    def __init__(self):
        self.network = DroneNetwork()
        self.algorithms = GraphAlgorithms()
        self.no_fly_zones = []  # For area-based restrictions (optional)
    
    # ==================== B1: IMPORT/EXPORT ====================
    
    def load_network(self, filepath: str) -> None:
        """B1: Load network from JSON file"""
        self.network.load_from_json(filepath)
        stats = self.network.get_statistics()
        print(f"\n✓ Network loaded successfully!")
        print(f"  Nodes: {stats['total_nodes']}")
        print(f"  Edges: {stats['total_edges']}")
        print(f"  Node types: {stats['node_types']}")
        print(f"  Restricted edges: {stats['restricted_edges']}")
    
    def save_network(self, filepath: str) -> None:
        """B1: Save network to JSON file"""
        self.network.save_to_json(filepath)
        print(f"✓ Network saved to {filepath}")
    
    def display_network_info(self) -> None:
        """Display detailed network statistics"""
        stats = self.network.get_statistics()
        print(f"\n{'='*70}")
        print(f"DRONE NETWORK STATISTICS")
        print(f"{'='*70}")
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Total Edges: {stats['total_edges']}")
        print(f"\nNode Types:")
        for ntype, count in stats['node_types'].items():
            print(f"  - {ntype}: {count}")
        print(f"\nEdge Properties:")
        print(f"  - Restricted (No-fly): {stats['restricted_edges']}")
        print(f"  - Bidirectional: {stats['bidirectional_edges']}")
        print(f"{'='*70}\n")
    
    # ==================== B2: NO-FLY ZONES ====================
    
    def mark_no_fly_zone(self, from_node: str, to_node: str) -> None:
        """B2: Mark corridor as no-fly zone"""
        if self.network.mark_restricted(from_node, to_node):
            print(f"✓ Corridor {from_node} → {to_node} marked as NO-FLY ZONE")
        else:
            print(f"✗ Corridor {from_node} → {to_node} not found")
    
    def remove_no_fly_zone(self, from_node: str, to_node: str) -> None:
        """B2: Remove no-fly zone restriction"""
        if self.network.unmark_restricted(from_node, to_node):
            print(f"✓ No-fly zone removed from {from_node} → {to_node}")
        else:
            print(f"✗ Restriction not found")
    
    def add_area_no_fly_zone(self, x: float, y: float, radius: float) -> None:
        """B2 Optional: Add area-based no-fly zone"""
        self.no_fly_zones.append({'x': x, 'y': y, 'radius': radius, 'type': 'circular'})
        print(f"✓ Area no-fly zone added at ({x}, {y}) with radius {radius}")
    
    def list_no_fly_zones(self) -> None:
        """B2: List all no-fly zones"""
        print(f"\n{'='*70}")
        print("NO-FLY ZONES")
        print(f"{'='*70}")
        
        # Corridor-based restrictions
        restricted_count = 0
        for edge in self.network.edges:
            if edge.restricted:
                print(f"  {edge.from_node} → {edge.to_node}")
                restricted_count += 1
        
        if restricted_count == 0:
            print("  No corridor restrictions")
        
        # Area-based restrictions
        if self.no_fly_zones:
            print(f"\nArea-based restrictions: {len(self.no_fly_zones)}")
            for zone in self.no_fly_zones:
                print(f"  Center: ({zone['x']}, {zone['y']}), Radius: {zone['radius']}")
        
        print(f"{'='*70}\n")
    
    # ==================== B3: MODIFY NETWORK ====================
    
    def add_charging_station(self, node_id: str, name: str, x: float, y: float) -> None:
        """B3: Add new charging station"""
        node = Node(node_id, name, "charging", x, y)
        self.network.add_node(node)
        print(f"✓ Charging station '{name}' added at ({x}, {y})")
    
    def add_delivery_point(self, node_id: str, name: str, x: float, y: float) -> None:
        """B3: Add new delivery point"""
        node = Node(node_id, name, "delivery", x, y)
        self.network.add_node(node)
        print(f"✓ Delivery point '{name}' added at ({x}, {y})")
    
    def add_corridor(self, from_node: str, to_node: str, energy: float, 
                     capacity: int, distance: float, bidirectional: bool = False) -> None:
        """B3: Add new flight corridor"""
        edge = Edge(from_node, to_node, energy, capacity, distance, bidirectional)
        self.network.add_edge(edge)
        direction = "bidirectional" if bidirectional else "one-way"
        print(f"✓ {direction} corridor added: {from_node} → {to_node}")
    
    def remove_corridor(self, from_node: str, to_node: str) -> None:
        """B3: Remove flight corridor"""
        if self.network.remove_edge(from_node, to_node):
            print(f"✓ Corridor {from_node} → {to_node} removed")
        else:
            print(f"✗ Corridor not found")
    
    def update_corridor_cost(self, from_node: str, to_node: str, new_energy: float) -> None:
        """B3: Update energy cost of corridor"""
        if self.network.update_energy_cost(from_node, to_node, new_energy):
            print(f"✓ Energy cost updated: {from_node} → {to_node} = {new_energy}")
        else:
            print(f"✗ Corridor not found")
    
    # ==================== F1: CHECK REACHABILITY ====================
    
    def check_reachability(self, start: str, end: str) -> bool:
        """F1: Check if destination is reachable from start"""
        reachable = self.algorithms.bfs_reachability(self.network, start, end)
        
        if reachable:
            print(f"\n✓ REACHABLE: {start} can reach {end}")
        else:
            print(f"\n✗ NOT REACHABLE: {start} cannot reach {end}")
        
        return reachable
    
    # ==================== F2: EFFICIENT FLIGHT ROUTES ====================
    
    def find_shortest_path(self, start: str, end: str, use_energy: bool = False) -> None:
        """F2: Find most efficient route using Dijkstra"""
        metric = "energy cost" if use_energy else "distance"
        distance, path = self.algorithms.dijkstra(self.network, start, end, use_energy)
        
        if path:
            print(f"\n✓ Most Efficient Route ({metric}):")
            print(f"  Path: {' → '.join(path)}")
            print(f"  Total {metric}: {distance:.2f}")
            print(f"  Number of hops: {len(path) - 1}")
            
            # Show node types along route
            node_types = [self.network.get_node(n).type for n in path]
            print(f"  Route profile: {' → '.join(node_types)}")
        else:
            print(f"\n✗ No route found between {start} and {end}")
    
    # ==================== F3: DELIVERY CAPACITY ====================
    
    def compute_delivery_capacity(self, source: str, sink: str) -> None:
        """F3: Calculate maximum delivery capacity using Ford-Fulkerson"""
        max_flow = self.algorithms.ford_fulkerson(self.network, source, sink)
        
        print(f"\n✓ Delivery Capacity Analysis:")
        print(f"  From: {source}")
        print(f"  To: {sink}")
        print(f"  Maximum capacity: {max_flow} drones/hour")
        
        if max_flow == 0:
            print(f"  ⚠ Warning: No available capacity (check for restrictions)")
        elif max_flow < 5:
            print(f"  ⚠ Warning: Low capacity - consider adding corridors")
    
    # ==================== F4: NETWORK RESILIENCE ====================
    
    def assess_resilience(self) -> None:
        """F4: Assess network resilience and identify bottlenecks"""
        analysis = self.algorithms.analyze_connectivity(self.network)
        
        if not analysis:
            print("\n✗ Network is empty")
            return
        
        print(f"\n{'='*70}")
        print("F4: NETWORK RESILIENCE ASSESSMENT")
        print(f"{'='*70}")
        print(f"Average connectivity: {analysis['average_connectivity']:.2f}")
        print(f"Minimum connectivity: {analysis['min_connectivity']}")
        print(f"Maximum connectivity: {analysis['max_connectivity']}")
        
        print(f"\n🔴 Critical Nodes (Bottlenecks):")
        if analysis['critical_nodes']:
            for node in analysis['critical_nodes'][:10]:
                node_obj = self.network.get_node(node)
                print(f"  - {node} ({node_obj.type}): {analysis['connectivity_map'][node]} connections")
        else:
            print("  None identified")
        
        if analysis['isolated_nodes']:
            print(f"\n⚠ Isolated Nodes: {', '.join(analysis['isolated_nodes'])}")
        
        # Resilience score
        if analysis['min_connectivity'] >= 3:
            score = "Excellent"
        elif analysis['min_connectivity'] >= 2:
            score = "Good"
        elif analysis['min_connectivity'] >= 1:
            score = "Fair"
        else:
            score = "Poor"
        
        print(f"\nResilience Score: {score}")
        
        # Find bridges (critical corridors)
        print(f"\n🔴 Critical Corridors (Bridges):")
        bridges = self.algorithms.find_bridges(self.network)
        if bridges:
            for from_node, to_node in bridges[:10]:
                print(f"  - {from_node} → {to_node}")
            print(f"  Total bridges: {len(bridges)}")
        else:
            print("  None identified (robust network)")
        
        print(f"{'='*70}\n")
    
    # ==================== F5: OPTIMIZE CHARGING STATIONS ====================
    
    def optimize_charging_stations(self, k: int = None) -> None:
        """F5: Optimize placement of k charging stations"""
        current_analysis = self.algorithms.analyze_charging_coverage(self.network)
        
        print(f"\n{'='*70}")
        print("F5: CHARGING STATION OPTIMIZATION")
        print(f"{'='*70}")
        
        print(f"\nCurrent Configuration:")
        print(f"  Charging stations: {current_analysis['charging_stations']}")
        print(f"  Delivery points: {current_analysis['delivery_points']}")
        print(f"  Average distance: {current_analysis['average_distance']:.2f} units")
        print(f"  Maximum distance: {current_analysis['max_distance']:.2f} units")
        print(f"  Coverage quality: {current_analysis['coverage_quality']}")
        
        if k is not None and k > 0:
            print(f"\nOptimal Placement for {k} Additional Stations:")
            optimal_locations = self.algorithms.optimize_charging_placement(self.network, k)
            
            for i, (x, y) in enumerate(optimal_locations, 1):
                print(f"  Station {i}: ({x:.2f}, {y:.2f})")
            
            print(f"\n💡 Recommendation: Add charging stations at these coordinates")
        
        print(f"{'='*70}\n")
    
    # ==================== F6: COMMUNICATION INFRASTRUCTURE ====================
    
    def compute_communication_network(self, use_energy: bool = False) -> None:
        """F6: Compute minimum cost communication network using Kruskal MST"""
        metric = "energy cost" if use_energy else "distance"
        mst_edges, total_cost = self.algorithms.kruskal_mst(self.network, use_energy)
        
        print(f"\n{'='*70}")
        print("F6: COMMUNICATION INFRASTRUCTURE (Minimum Spanning Tree)")
        print(f"{'='*70}")
        print(f"Optimization metric: {metric}")
        print(f"Total network nodes: {len(self.network.nodes)}")
        print(f"Communication links needed: {len(mst_edges)}")
        print(f"Total setup cost: {total_cost:.2f}")
        
        print(f"\nCommunication Links:")
        for i, (from_node, to_node, cost) in enumerate(mst_edges[:15], 1):
            from_type = self.network.get_node(from_node).type
            to_type = self.network.get_node(to_node).type
            print(f"  {i}. {from_node} ({from_type}) ↔ {to_node} ({to_type}): {cost:.2f}")
        
        if len(mst_edges) > 15:
            print(f"  ... and {len(mst_edges) - 15} more links")
        
        # Calculate average cost
        if mst_edges:
            avg_cost = total_cost / len(mst_edges)
            print(f"\nAverage link cost: {avg_cost:.2f}")
        
        print(f"{'='*70}\n")
    
    def setup_communication_range_analysis(self, comm_range: float = 30) -> None:
        """F6: Alternative - Analyze direct communication based on range"""
        connections = 0
        nodes_list = list(self.network.nodes.values())
        connection_list = []
        
        for i, node1 in enumerate(nodes_list):
            for node2 in nodes_list[i+1:]:
                dx = node1.x - node2.x
                dy = node1.y - node2.y
                distance = (dx * dx + dy * dy) ** 0.5
                
                if distance <= comm_range:
                    connections += 1
                    connection_list.append((node1.id, node2.id, distance))
        
        print(f"\n{'='*70}")
        print("F6: COMMUNICATION RANGE ANALYSIS")
        print(f"{'='*70}")
        print(f"Communication range: {comm_range} units")
        print(f"Total nodes: {len(nodes_list)}")
        print(f"Direct connections possible: {connections}")
        
        if nodes_list:
            avg_connections = (2 * connections) / len(nodes_list)
            print(f"Average connections per node: {avg_connections:.2f}")
        
        # Check connectivity
        min_connections_needed = len(nodes_list) - 1
        if connections >= min_connections_needed:
            print(f"✓ Network can be fully connected")
        else:
            print(f"⚠ Warning: May not be fully connected (need {min_connections_needed}, have {connections})")
        
        print(f"{'='*70}\n")