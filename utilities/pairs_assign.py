import csv
import numpy as np
import itertools
from scipy.optimize import linear_sum_assignment

def read_preferences(filename):
    """
    Read student preferences from a CSV file.
    
    Expected format:
    - First row: header with "Student" and project titles
    - Subsequent rows: student name and their preferences (e.g., "1st", "2nd", etc.)
    
    Returns:
    - students: list of student names
    - projects: list of project titles
    - preferences: dictionary mapping student names to dictionaries of project preferences
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        
        # Extract project titles (skip first column which is student name)
        projects = header[1:]
        
        students = []
        preferences = {}
        
        for row in reader:
            student_name = row[0]
            students.append(student_name)
            
            student_prefs = {}
            for i, pref in enumerate(row[1:]):
                # Convert "1st", "2nd", etc. to integers
                if pref:  # Skip empty cells
                    rank = int(pref.replace("st", "").replace("nd", "").replace("rd", "").replace("th", ""))
                    student_prefs[projects[i]] = rank
                else:
                    # Assign a large value for no preference
                    student_prefs[projects[i]] = 999
            
            preferences[student_name] = student_prefs
    
    return students, projects, preferences

def generate_all_possible_pairings(students, preferences, top_n=3):
    """
    Generate all possible ways to divide students into pairs, but only include pairs
    where the students share at least one project in their top N preferences.
    
    Parameters:
    - students: List of student names
    - preferences: Dictionary mapping student names to project preferences
    - top_n: Number of top preferences to consider for compatibility (default: 5)
    
    Returns:
    - A list of pairings, where each pairing is a list of pairs,
      and each pair is a tuple of two student names.
    """
    n = len(students)
    if n % 2 != 0:
        raise ValueError("Number of students must be even to form pairs")
    
    # First, determine which pairs of students are compatible (share preferences)
    compatible_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            student1 = students[i]
            student2 = students[j]
            
            # Get top N preferences for each student
            top_prefs1 = set([proj for proj, rank in sorted(preferences[student1].items(), 
                                                          key=lambda x: x[1])[:top_n]])
            top_prefs2 = set([proj for proj, rank in sorted(preferences[student2].items(), 
                                                          key=lambda x: x[1])[:top_n]])
            
            # Check if they share any preferred projects
            if top_prefs1.intersection(top_prefs2):
                compatible_pairs.append((student1, student2))
    
    print(f"Found {len(compatible_pairs)} compatible student pairs out of {n*(n-1)//2} possible pairs")
    
    # Build an adjacency list representation for compatible students
    adjacency = {student: [] for student in students}
    for s1, s2 in compatible_pairs:
        adjacency[s1].append(s2)
        adjacency[s2].append(s1)
    
    # Generate all possible pairings using only compatible pairs
    all_pairings = []
    get_valid_pairings(students.copy(), [], all_pairings, adjacency)
    
    return all_pairings

def get_valid_pairings(available_students, current_pairing, all_pairings, adjacency):
    """
    Recursive helper function to generate all possible pairings of students,
    considering only compatible student pairs.
    
    Parameters:
    - available_students: List of students not yet paired
    - current_pairing: List of pairs already formed
    - all_pairings: Output list where valid complete pairings are collected
    - adjacency: Dictionary mapping students to lists of compatible partners
    """
    # If no students left, we found a complete valid pairing
    if not available_students:
        all_pairings.append(current_pairing.copy())
        return
    
    # Take the first available student
    first = available_students[0]
    
    # Try to pair them with each compatible student
    for partner in adjacency[first]:
        if partner in available_students:  # Make sure partner is still available
            # Form the pair
            pair = (first, partner) if first < partner else (partner, first)
            
            # Create new lists for the recursive call
            new_available = available_students.copy()
            new_available.remove(first)
            new_available.remove(partner)
            
            new_pairing = current_pairing.copy()
            new_pairing.append(pair)
            
            # Recurse with the new state
            get_valid_pairings(new_available, new_pairing, all_pairings, adjacency)

def calculate_cost_matrix(student_pairs, projects, preferences):
    """
    Calculate the cost matrix for a given set of student pairs and projects.
    
    Returns:
    - cost_matrix: 2D array where cost_matrix[i][j] is the cost of assigning pair i to project j
    """
    # Find max rank for scaling
    max_rank = 0
    for student in preferences:
        for project in projects:
            max_rank = max(max_rank, preferences[student].get(project, 0))
    
    # Create cost matrix
    cost_matrix = np.zeros((len(student_pairs), len(projects)))
    
    for i, pair in enumerate(student_pairs):
        for j, project in enumerate(projects):
            rank1 = preferences[pair[0]].get(project, max_rank * 2)
            rank2 = preferences[pair[1]].get(project, max_rank * 2)
            
            # Calculate cost for this pair and project
            # We prioritize by sum of ranks, then by difference between ranks
            sum_ranks = rank1 + rank2
            diff_ranks = abs(rank1 - rank2)
            
            # Combine into single cost value where:
            # Lower sum always beats higher sum
            # With equal sums, lower difference is better
            cost = sum_ranks * (max_rank * 2) + diff_ranks
            
            cost_matrix[i, j] = cost
    
    return cost_matrix

def evaluate_pairing(student_pairs, projects, preferences):
    """
    Evaluate a specific pairing of students by running the Hungarian algorithm.
    
    Returns:
    - assignments: dictionary mapping project names to lists of assigned students
    - total_cost: total cost of the assignment (lower is better)
    """
    # Calculate cost matrix for this pairing
    cost_matrix = calculate_cost_matrix(student_pairs, projects, preferences)
    
    # Extend the cost matrix if necessary to make it square
    num_pairs = len(student_pairs)
    num_projects = len(projects)
    
    if num_pairs > num_projects:
        # Add dummy projects with high cost
        dummy_cols = np.full((num_pairs, num_pairs - num_projects), np.max(cost_matrix) * 10)
        extended_cost_matrix = np.hstack((cost_matrix, dummy_cols))
    else:
        extended_cost_matrix = cost_matrix
    
    # Apply the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(extended_cost_matrix)
    
    # Process the assignments
    assignments = {}
    costs = []
    
    for pair_idx, project_idx in zip(row_indices, col_indices):
        # Skip dummy projects
        if project_idx >= num_projects:
            continue
        
        project = projects[project_idx]
        pair = student_pairs[pair_idx]
        
        assignments[project] = list(pair)
        
        # Add the cost for this assignment
        costs.append(cost_matrix[pair_idx, project_idx])
    
    return assignments, sum(costs)

def find_optimal_assignment(students, projects, preferences, top_n=3):
    """
    Find the optimal pairing of students and assignment to projects.
    
    The approach:
    1. Generate possible ways to pair up students where pairs share preferences
    2. For each pairing, calculate the cost matrix and run the Hungarian algorithm
    3. Select the pairing that results in the lowest total cost
    
    Parameters:
    - students: List of student names
    - projects: List of project names
    - preferences: Dictionary of student preferences
    - top_n: Number of top preferences to consider for compatibility (default: 5)
    
    Returns:
    - best_assignments: dictionary mapping project names to lists of assigned students
    - best_pairing: the best pairing of students
    """
    from tqdm import tqdm
    all_pairings = generate_all_possible_pairings(students, preferences, top_n)
    
    if not all_pairings:
        print("Warning: No valid pairings found with the current compatibility criteria.")
        print(f"Try increasing the top_n parameter (currently {top_n}) or relaxing the compatibility requirements.")
        return {}, []
    
    print(f"Generated {len(all_pairings)} possible pairings of students")
    
    best_assignments = None
    best_cost = float('inf')
    best_pairing = None
    
    for pairing in tqdm(all_pairings):
        # For large problem sizes, it helps to show progress
        
        assignments, total_cost = evaluate_pairing(pairing, projects, preferences)
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_assignments = assignments
            best_pairing = pairing
    
    return best_assignments, best_pairing

def calculate_assignment_costs(assignments, preferences):
    """
    Calculate the preference ranks for each assignment.
    
    Returns:
    - assignment_costs: dictionary mapping project names to tuples of preference ranks
    """
    assignment_costs = {}
    
    for project, students in assignments.items():
        student1, student2 = students
        rank1 = preferences[student1][project]
        rank2 = preferences[student2][project]
        assignment_costs[project] = (rank1, rank2)
    
    return assignment_costs

def print_assignments(assignments, assignment_costs, students, projects):
    """Print the assignments in a readable format."""
    if not assignments:
        print("\nNo projects were assigned.")
        return
        
    print("\nProject Assignments:")
    print("=" * 60)
    
    # Sort projects by assignment quality (sum of ranks)
    sorted_projects = sorted(
        assignments.keys(), 
        key=lambda p: sum(assignment_costs[p])
    )
    
    for project in sorted_projects:
        assigned_students = assignments[project]
        print(f"\nProject: {project}")
        print("-" * 40)
        
        for student in assigned_students:
            rank = assignment_costs[project][assigned_students.index(student)]
            suffix = 'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'
            print(f"  {student} (Preference: {rank}{suffix})")
        
        rank1, rank2 = assignment_costs[project]
        print(f"  Assignment Quality: ({rank1}, {rank2})")
    
    # List unassigned projects
    assigned_projects = set(assignments.keys())
    unassigned_projects = set(projects) - assigned_projects
    
    if unassigned_projects:
        print("\nUnassigned Projects:")
        print("-" * 40)
        for project in sorted(unassigned_projects):
            print(f"  {project}")
    
    # List unassigned students
    assigned_students = set()
    for students_list in assignments.values():
        assigned_students.update(students_list)
    
    unassigned_students = set(students) - assigned_students
    
    if unassigned_students:
        print("\nUnassigned Students:")
        print("-" * 40)
        for student in sorted(unassigned_students):
            print(f"  {student}")
    
    print("\n" + "=" * 60)

def assign_projects(filename, top_n=3):
    """
    Main function to assign students to projects.
    
    Parameters:
    - filename: CSV file with student preferences
    - top_n: Number of top preferences to consider for compatibility (default: 5)
    
    Returns:
    - assignments: dictionary mapping project names to lists of assigned students
    - assignment_costs: dictionary mapping project names to their assignment cost
    - students: list of student names
    - projects: list of project titles
    """
    # Read data from CSV
    students, projects, preferences = read_preferences(filename)
    
    # Check if we have an even number of students
    if len(students) % 2 != 0:
        raise ValueError(f"Number of students ({len(students)}) must be even to form pairs.")
    
    print(f"Processing preferences for {len(students)} students and {len(projects)} projects")
    print(f"Only considering student pairs sharing at least one project in their top {top_n} preferences")
    
    # Find the optimal assignment
    assignments, best_pairing = find_optimal_assignment(students, projects, preferences, top_n)
    
    if not assignments:
        print("Failed to find valid assignments. Try adjusting parameters.")
        return {}, {}, students, projects
    
    # Calculate the assignment costs
    assignment_costs = calculate_assignment_costs(assignments, preferences)
    
    return assignments, assignment_costs, students, projects

def main():
    import sys
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Assign students to projects based on preferences.')
    parser.add_argument('preferences_csv', help='CSV file containing student preferences')
    parser.add_argument('--top', type=int, default=3, 
                       help='Number of top preferences to consider for compatibility (default: 5)')
    
    args = parser.parse_args()
    
    try:
        # Assign projects
        assignments, assignment_costs, students, projects = assign_projects(args.preferences_csv, args.top)
        
        # Print the results
        print_assignments(assignments, assignment_costs, students, projects)
        
        # Calculate overall satisfaction
        if assignment_costs:
            total_cost = sum(sum(costs) for costs in assignment_costs.values())
            num_assigned_students = sum(len(students_list) for students_list in assignments.values())
            avg_preference = total_cost / num_assigned_students
            
            print(f"\nOverall Statistics:")
            print(f"Students: {len(students)} total, {num_assigned_students} assigned")
            print(f"Projects: {len(projects)} total, {len(assignments)} assigned")
            print(f"Average Preference Rank: {avg_preference:.2f}")
            
            # Count perfect assignments (1,1)
            perfect_count = sum(1 for costs in assignment_costs.values() if costs == (1, 1))
            print(f"Projects with Perfect Assignment (1,1): {perfect_count} out of {len(assignments)}")
            
            # Count by preference type
            preference_counts = {}
            for cost_pair in assignment_costs.values():
                sorted_pair = tuple(sorted(cost_pair))
                if sorted_pair in preference_counts:
                    preference_counts[sorted_pair] += 1
                else:
                    preference_counts[sorted_pair] = 1
            
            print("\nAssignment Types:")
            for pair, count in sorted(preference_counts.items()):
                print(f"  ({pair[0]},{pair[1]}): {count} projects")
        else:
            print("\nNo assignments were made. Try increasing the top-n parameter.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
