"""Graph algorithms - Implements F1, F2, F3, F4, F6"""
import heapq  # PRIORITY QUEUE (MIN-HEAP)
from typing import Dict, List, Tuple, Optional
from collections import deque  # QUEUE


class GraphAlgorithms:
    """
    Implements all required algorithms with proper data structures
    
    DATA STRUCTURES USED:
    - Priority Queue (heapq): Dijkstra's algorithm
    - Queue (deque): BFS, Ford-Fulkerson
    - Hash Map (dict): Distance tracking, parent tracking
    - Hash Set (set): Visited nodes
    - Union-Find: Kruskal's MST
    """
    
    # ==================== F2: EFFICIENT FLIGHT ROUTES ====================
    
    @staticmethod
    def dijkstra(network, start: str, end: str, 
                 use_energy: bool = False, avoid_restricted: bool = True) -> Tuple[float, Optional[List[str]]]:
        """
        F2: Dijkstra's shortest path algorithm
        
        DATA STRUCTURES:
        - Priority Queue (heapq): Extract minimum distance node in O(log n)
        - Hash Map (dict): Store distances in O(1)
        - Hash Set (set): Track visited nodes in O(1)
        
        Time Complexity: O((V + E) log V)
        """
        if start not in network.nodes or end not in network.nodes:
            return float('inf'), None
        
        # Initialize with Hash Maps
        distances = {node_id: float('inf') for node_id in network.nodes}
        distances[start] = 0
        previous = {node_id: None for node_id in network.nodes}
        
        # PRIORITY QUEUE (MIN-HEAP) - O(log n) operations
        pq = [(0, start)]  # (distance, node)
        visited = set()  # HASH SET - O(1) membership check
        
        while pq:
            # Extract minimum - O(log n)
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == end:
                break
            
            # Explore neighbors
            for neighbor, edge in network.get_neighbors(current):
                # B2: Avoid no-fly zones
                if avoid_restricted and edge.restricted:
                    continue
                
                if neighbor not in visited:
                    # Choose metric: energy or distance
                    edge_cost = edge.energy_cost if use_energy else edge.distance
                    new_dist = current_dist + edge_cost
                    
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current
                        # Insert into priority queue - O(log n)
                        heapq.heappush(pq, (new_dist, neighbor))
        
        # Reconstruct path
        if distances[end] == float('inf'):
            return float('inf'), None
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        
        path.reverse()
        return distances[end], path
    
    # ==================== F1: CHECK REACHABILITY ====================
    
    @staticmethod
    def bfs_reachability(network, start: str, end: str, avoid_restricted: bool = True) -> bool:
        """
        F1: Check if end is reachable from start using BFS
        
        DATA STRUCTURES:
        - Queue (deque): FIFO for BFS - O(1) enqueue/dequeue
        - Hash Set (set): Track visited nodes - O(1) membership
        
        Time Complexity: O(V + E)
        """
        if start not in network.nodes or end not in network.nodes:
            return False
        
        visited = set([start])  # HASH SET
        queue = deque([start])  # QUEUE
        
        while queue:
            current = queue.popleft()  # O(1) dequeue
            
            if current == end:
                return True
            
            for neighbor, edge in network.get_neighbors(current):
                # B2: Respect no-fly zones
                if avoid_restricted and edge.restricted:
                    continue
                    
                if neighbor not in visited:
                    visited.add(current)
                    queue.append(neighbor)  # O(1) enqueue
        
        return False
    
    # ==================== F6: COMMUNICATION INFRASTRUCTURE ====================
    
    @staticmethod
    def kruskal_mst(network, use_energy: bool = False) -> Tuple[List[Tuple[str, str, float]], float]:
        """
        F6: Kruskal's Minimum Spanning Tree for communication network
        
        DATA STRUCTURES:
        - Union-Find (custom): Detect cycles in O(α(n)) amortized
        - Hash Map (dict): Parent and rank tracking
        
        Time Complexity: O(E log E)
        """
        # UNION-FIND DATA STRUCTURE
        parent = {node_id: node_id for node_id in network.nodes}
        rank = {node_id: 0 for node_id in network.nodes}
        
        def find(node):
            """Find with path compression - O(α(n)) amortized"""
            if parent[node] != node:
                parent[node] = find(parent[node])  # Path compression
            return parent[node]
        
        def union(node1, node2):
            """Union by rank - O(α(n)) amortized"""
            root1 = find(node1)
            root2 = find(node2)
            
            if root1 == root2:
                return False  # Cycle detected
            
            # Union by rank
            if rank[root1] < rank[root2]:
                parent[root1] = root2
            elif rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root2] = root1
                rank[root1] += 1
            
            return True
        
        # Create edge list with costs
        edge_list = []
        for edge in network.edges:
            if edge.restricted:
                continue  # Skip restricted corridors
            cost = edge.energy_cost if use_energy else edge.distance
            edge_list.append((edge.from_node, edge.to_node, cost))
        
        # Sort edges by cost - O(E log E)
        sorted_edges = sorted(edge_list, key=lambda e: e[2])
        
        mst_edges = []
        total_cost = 0
        
        # Build MST
        for from_node, to_node, cost in sorted_edges:
            if union(from_node, to_node):
                mst_edges.append((from_node, to_node, cost))
                total_cost += cost
                
                # Early termination: MST has V-1 edges
                if len(mst_edges) == len(network.nodes) - 1:
                    break
        
        return mst_edges, total_cost
    
    # ==================== F3: DELIVERY CAPACITY ====================
    
    @staticmethod
    def ford_fulkerson(network, source: str, sink: str) -> int:
        """
        F3: Ford-Fulkerson (Edmonds-Karp) for maximum flow
        
        DATA STRUCTURES:
        - Queue (deque): BFS for augmenting paths - O(1) operations
        - Hash Map (dict): Residual graph - O(1) capacity lookup
        - Hash Set (set): Visited tracking - O(1) membership
        
        Time Complexity: O(VE²)
        """
        if source not in network.nodes or sink not in network.nodes:
            return 0
        
        # RESIDUAL GRAPH using nested Hash Maps
        residual = {}
        for node_id in network.nodes:
            residual[node_id] = {}
        
        # Initialize residual capacities
        for edge in network.edges:
            if edge.restricted:  # B2: Skip no-fly zones
                continue
                
            residual[edge.from_node][edge.to_node] = edge.capacity
            
            # Initialize backward edge
            if edge.to_node not in residual:
                residual[edge.to_node] = {}
            if edge.from_node not in residual[edge.to_node]:
                residual[edge.to_node][edge.from_node] = 0
            
            # Bidirectional edges
            if edge.bidirectional:
                residual[edge.to_node][edge.from_node] = edge.capacity
        
        def bfs_find_path(source, sink, parent):
            """Find augmenting path using BFS (Edmonds-Karp variant)"""
            visited = set([source])  # HASH SET
            queue = deque([source])  # QUEUE
            
            while queue:
                u = queue.popleft()  # O(1) dequeue
                
                for v in residual.get(u, {}):
                    if v not in visited and residual[u][v] > 0:
                        visited.add(v)
                        parent[v] = u
                        
                        if v == sink:
                            return True
                        
                        queue.append(v)  # O(1) enqueue
            
            return False
        
        max_flow = 0
        parent = {}
        
        # Find augmenting paths
        while bfs_find_path(source, sink, parent):
            # Find bottleneck capacity
            path_flow = float('inf')
            s = sink
            
            while s != source:
                path_flow = min(path_flow, residual[parent[s]][s])
                s = parent[s]
            
            # Update residual capacities
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= path_flow  # Forward edge
                residual[v][u] += path_flow  # Backward edge
                v = parent[v]
            
            max_flow += path_flow
            parent = {}
        
        return max_flow
    
    # ==================== F4: NETWORK RESILIENCE ====================
    
    @staticmethod
    def analyze_connectivity(network) -> Dict:
        """
        F4: Analyze network resilience and identify bottlenecks
        
        Returns critical nodes with minimum connectivity
        """
        connectivity = {node_id: 0 for node_id in network.nodes}
        
        # Count connections per node
        for edge in network.edges:
            if not edge.restricted:
                connectivity[edge.from_node] += 1
                connectivity[edge.to_node] += 1
                if edge.bidirectional:
                    connectivity[edge.from_node] += 1
                    connectivity[edge.to_node] += 1
        
        if not connectivity:
            return {}
        
        avg_connectivity = sum(connectivity.values()) / len(connectivity)
        min_connectivity = min(connectivity.values())
        max_connectivity = max(connectivity.values())
        
        # F4: Identify critical nodes (bottlenecks)
        critical_nodes = [
            node_id for node_id, conn in connectivity.items()
            if conn == min_connectivity
        ]
        
        isolated = [node_id for node_id, conn in connectivity.items() if conn == 0]
        
        return {
            'average_connectivity': avg_connectivity,
            'min_connectivity': min_connectivity,
            'max_connectivity': max_connectivity,
            'critical_nodes': critical_nodes,  # F4: Bottlenecks
            'isolated_nodes': isolated,
            'connectivity_map': connectivity
        }
    
    @staticmethod
    def find_bridges(network) -> List[Tuple[str, str]]:
        """
        F4: Find bridges (edges whose removal disconnects the network)
        
        Uses Tarjan's algorithm for bridge finding
        Time Complexity: O(V + E)
        """
        visited = set()
        disc = {}
        low = {}
        parent = {}
        bridges = []
        time = [0]
        
        def dfs(u):
            visited.add(u)
            disc[u] = low[u] = time[0]
            time[0] += 1
            
            for v, edge in network.get_neighbors(u):
                if edge.restricted:
                    continue
                    
                if v not in visited:
                    parent[v] = u
                    dfs(v)
                    
                    low[u] = min(low[u], low[v])
                    
                    # F4: Bridge condition
                    if low[v] > disc[u]:
                        bridges.append((u, v))
                
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])
        
        for node_id in network.nodes:
            if node_id not in visited:
                dfs(node_id)
        
        return bridges
    
    # ==================== F5: OPTIMIZE CHARGING STATION PLACEMENT ====================
    
    @staticmethod
    def optimize_charging_placement(network, k: int = 1) -> List[Tuple[float, float]]:
        """
        F5: K-means clustering to optimize k charging station placements
        
        Goal: Minimize average distance from delivery points to nearest charging station
        
        Algorithm: K-means clustering on delivery point coordinates
        Time Complexity: O(n * k * iterations)
        """
        import random
        
        # Get all delivery points
        delivery_nodes = [n for n in network.nodes.values() if n.type == 'delivery']
        
        if not delivery_nodes or k <= 0:
            return []
        
        # Initialize centroids randomly from delivery points
        centroids = random.sample([(n.x, n.y) for n in delivery_nodes], min(k, len(delivery_nodes)))
        
        max_iterations = 100
        for iteration in range(max_iterations):
            # Assign each delivery point to nearest centroid
            clusters = [[] for _ in range(len(centroids))]
            
            for node in delivery_nodes:
                min_dist = float('inf')
                closest_cluster = 0
                
                for i, (cx, cy) in enumerate(centroids):
                    dist = ((node.x - cx) ** 2 + (node.y - cy) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_cluster = i
                
                clusters[closest_cluster].append((node.x, node.y))
            
            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    avg_x = sum(x for x, y in cluster) / len(cluster)
                    avg_y = sum(y for x, y in cluster) / len(cluster)
                    new_centroids.append((avg_x, avg_y))
                else:
                    # Keep old centroid if cluster is empty
                    new_centroids.append(centroids[len(new_centroids)])
            
            # Check convergence
            if new_centroids == centroids:
                break
            
            centroids = new_centroids
        
        return centroids
    
    @staticmethod
    def analyze_charging_coverage(network) -> Dict:
        """
        F5: Analyze current charging station coverage
        
        Returns metrics about charging station placement quality
        """
        delivery_nodes = [n for n in network.nodes.values() if n.type == 'delivery']
        charging_nodes = [n for n in network.nodes.values() if n.type == 'charging']
        
        if not delivery_nodes or not charging_nodes:
            return {
                'average_distance': float('inf'),
                'max_distance': float('inf'),
                'coverage_quality': 'Poor - Missing nodes'
            }
        
        # Calculate distance from each delivery point to nearest charging station
        distances = []
        coverage_map = {}
        
        for delivery in delivery_nodes:
            min_dist = float('inf')
            nearest_station = None
            
            for charging in charging_nodes:
                dx = delivery.x - charging.x
                dy = delivery.y - charging.y
                dist = (dx * dx + dy * dy) ** 0.5
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_station = charging.id
            
            distances.append(min_dist)
            coverage_map[delivery.id] = {
                'nearest_station': nearest_station,
                'distance': min_dist
            }
        
        avg_distance = sum(distances) / len(distances)
        max_distance = max(distances)
        
        # Quality assessment
        if max_distance < 30:
            quality = "Excellent"
        elif max_distance < 50:
            quality = "Good"
        elif max_distance < 80:
            quality = "Fair"
        else:
            quality = "Poor - Add more stations"
        
        return {
            'charging_stations': len(charging_nodes),
            'delivery_points': len(delivery_nodes),
            'average_distance': avg_distance,
            'max_distance': max_distance,
            'coverage_quality': quality,
            'coverage_map': coverage_map
        }