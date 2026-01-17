"""Main CLI application with all functionalities"""
from src.network_planner import NetworkPlanner


def print_header():
    """Print application header"""
    print("\n" + "="*75)
    print(" " * 20 + "DRONE NETWORK PLANNING SYSTEM")
    print(" " * 25 + "Nova Schilda City")
    print("="*75)


def print_menu():
    """Print main menu with all functionalities"""
    print("\n📋 MAIN MENU:")
    print("-" * 75)
    print("  BASIC OPERATIONS (B1-B3)")
    print("  [1]  B1: Load Network from JSON")
    print("  [2]  B1: Display Network Information")
    print("  [3]  B1: Save Network to JSON")
    print("  [4]  B2: Mark No-Fly Zone")
    print("  [5]  B2: Remove No-Fly Zone")
    print("  [6]  B2: List All No-Fly Zones")
    print("  [7]  B3: Add Charging Station")
    print("  [8]  B3: Add Flight Corridor")
    print("  [9]  B3: Update Corridor Energy Cost")
    print("")
    print("  ANALYSIS FUNCTIONS (F1-F6)")
    print("  [10] F1: Check Reachability (BFS)")
    print("  [11] F2: Find Shortest Path - Distance (Dijkstra)")
    print("  [12] F2: Find Shortest Path - Energy (Dijkstra)")
    print("  [13] F3: Compute Delivery Capacity (Ford-Fulkerson)")
    print("  [14] F4: Assess Network Resilience")
    print("  [15] F5: Optimize Charging Station Placement")
    print("  [16] F6: Compute Communication Network (Kruskal MST)")
    print("  [17] F6: Communication Range Analysis")
    print("")
    print("  [0]  Exit")
    print("-" * 75)


def main():
    """Main application loop"""
    planner = NetworkPlanner()
    print_header()
    
    print("\n🚁 Welcome to the Drone Network Planning System!")
    print("   Comprehensive tool for optimizing autonomous delivery networks")
    
    while True:
        print_menu()
        choice = input("\n➤ Enter your choice: ").strip()
        
        try:
            # ==================== B1: IMPORT/EXPORT ====================
            if choice == '1':
                print("\nAvailable test files:")
                print("  1. data/drone_testdata_1.json (Small - 7 nodes)")
                print("  2. data/drone_testdata_2.json (Large - 45 nodes)")
                file_choice = input("Select file (1/2) or enter path: ").strip()
                
                if file_choice == '1':
                    filepath = "data/drone_testdata_1.json"
                elif file_choice == '2':
                    filepath = "data/drone_testdata_2.json"
                else:
                    filepath = file_choice
                
                planner.load_network(filepath)
            
            elif choice == '2':
                planner.display_network_info()
            
            elif choice == '3':
                filepath = input("Enter output file path [data/output.json]: ").strip()
                if not filepath:
                    filepath = "data/output.json"
                planner.save_network(filepath)
            
            # ==================== B2: NO-FLY ZONES ====================
            elif choice == '4':
                from_node = input("From node: ").strip()
                to_node = input("To node: ").strip()
                planner.mark_no_fly_zone(from_node, to_node)
            
            elif choice == '5':
                from_node = input("From node: ").strip()
                to_node = input("To node: ").strip()
                planner.remove_no_fly_zone(from_node, to_node)
            
            elif choice == '6':
                planner.list_no_fly_zones()
            
            # ==================== B3: MODIFY NETWORK ====================
            elif choice == '7':
                node_id = input("Station ID (e.g., C11): ").strip()
                name = input("Station name: ").strip()
                x = float(input("X coordinate: ").strip())
                y = float(input("Y coordinate: ").strip())
                planner.add_charging_station(node_id, name, x, y)
            
            elif choice == '8':
                from_node = input("From node: ").strip()
                to_node = input("To node: ").strip()
                energy = float(input("Energy cost: ").strip())
                capacity = int(input("Capacity: ").strip())
                distance = float(input("Distance: ").strip())
                bidirectional = input("Bidirectional? (y/n): ").strip().lower() == 'y'
                planner.add_corridor(from_node, to_node, energy, capacity, distance, bidirectional)
            
            elif choice == '9':
                from_node = input("From node: ").strip()
                to_node = input("To node: ").strip()
                new_energy = float(input("New energy cost: ").strip())
                planner.update_corridor_cost(from_node, to_node, new_energy)
            
            # ==================== F1: REACHABILITY ====================
            elif choice == '10':
                start = input("Start node: ").strip()
                end = input("End node: ").strip()
                planner.check_reachability(start, end)
            
            # ==================== F2: SHORTEST PATH ====================
            elif choice == '11':
                start = input("Start node: ").strip()
                end = input("End node: ").strip()
                planner.find_shortest_path(start, end, use_energy=False)
            
            elif choice == '12':
                start = input("Start node: ").strip()
                end = input("End node: ").strip()
                planner.find_shortest_path(start, end, use_energy=True)
            
            # ==================== F3: DELIVERY CAPACITY ====================
            elif choice == '13':
                source = input("Source node (e.g., HUB): ").strip()
                sink = input("Sink node (e.g., D14): ").strip()
                planner.compute_delivery_capacity(source, sink)
            
            # ==================== F4: RESILIENCE ====================
            elif choice == '14':
                planner.assess_resilience()
            
            # ==================== F5: CHARGING OPTIMIZATION ====================
            elif choice == '15':
                k_input = input("Number of additional stations to optimize (Enter for analysis only): ").strip()
                k = int(k_input) if k_input else None
                planner.optimize_charging_stations(k)
            
            # ==================== F6: COMMUNICATION ====================
            elif choice == '16':
                metric = input("Optimize for (d)istance or (e)nergy? [d]: ").strip().lower()
                use_energy = (metric == 'e')
                planner.compute_communication_network(use_energy)
            
            elif choice == '17':
                comm_range = input("Communication range [30]: ").strip()
                comm_range = float(comm_range) if comm_range else 30
                planner.setup_communication_range_analysis(comm_range)
            
            # ==================== EXIT ====================
            elif choice == '0':
                print("\n" + "="*75)
                print("Thank you for using the Drone Network Planning System!")
                print("Safe flights! 🚁")
                print("="*75 + "\n")
                break
            
            else:
                print("✗ Invalid choice. Please try again.")
        
        except FileNotFoundError as e:
            print(f"✗ Error: File not found - {e}")
        except ValueError as e:
            print(f"✗ Error: Invalid input - {e}")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n⏎ Press Enter to continue...")


if __name__ == "__main__":
    main()