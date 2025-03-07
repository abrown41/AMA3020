import pandas as pd
import numpy as np
import re
from scipy.optimize import linear_sum_assignment

def parse_preference_csv(file_path):
    """
    Parse a CSV file with student project preferences and convert it to the format 
    required by the project assignment algorithm.
    
    The expected CSV format:
    - First row contains project titles
    - First column contains student names
    - Cells contain preference designations like '1st', '2nd', '3rd', etc.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    tuple
        (student_preferences, available_projects)
        - student_preferences: dict mapping student names to lists of projects in preference order
        - available_projects: list of all available project titles
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract student names from the first column
    # Assuming the first column contains student names
    student_column = df.columns[0]
    students = df[student_column].tolist()
    
    # Extract project titles from the header (excluding the first column which has student names)
    projects = df.columns[1:].tolist()
    
    # Initialize the dictionary for student preferences
    student_preferences = {}
    
    # Regular expression to match preference designations like '1st', '2nd', '3rd', etc.
    # and extract the numeric part
    preference_pattern = re.compile(r'(\d+)(?:st|nd|rd|th)')
    
    # Process each student row
    for index, row in df.iterrows():
        student_name = row[student_column]
        preferences = []
        
        # Create a list to store (preference_rank, project) tuples
        ranked_projects = []
        
        # Go through each project column
        for project in projects:
            cell_value = str(row[project]).strip()
            
            # Skip empty cells or cells with NaN
            if cell_value.lower() in ('', 'nan', 'none', 'null'):
                continue
            
            # Try to extract the preference rank
            match = preference_pattern.search(cell_value.lower())
            if match:
                rank = int(match.group(1))
                ranked_projects.append((rank, project))
            else:
                # If the cell doesn't match the pattern but contains data,
                # log a warning but don't stop processing
                print(f"Warning: Could not parse preference '{cell_value}' for student '{student_name}' and project '{project}'")
        
        # Sort projects by preference rank
        ranked_projects.sort()
        
        # Extract just the project names in ranked order
        preferences = [project for _, project in ranked_projects]
        
        # Store in the dictionary
        student_preferences[student_name] = preferences
    
    return student_preferences, projects

def assign_projects(student_preferences, available_projects):
    """
    Assigns projects to students using the Hungarian algorithm based on ranked preferences.
    
    Parameters:
    -----------
    student_preferences : dict
        Dictionary where keys are student names and values are lists of project names
        in order of preference (most preferred first)
    
    available_projects : list
        List of all available project names
    
    Returns:
    --------
    dict
        Dictionary mapping student names to their assigned project
    """
    # Create a cost matrix for the Hungarian algorithm
    # Higher costs for lower preferences, ensuring the algorithm minimizes the cost
    num_students = len(student_preferences)
    num_projects = len(available_projects)
    
    # Ensure we have at least as many projects as students
    if num_projects < num_students:
        raise ValueError(f"Not enough projects ({num_projects}) for all students ({num_students})")
    
    # Create a mapping of project names to indices
    project_to_index = {project: i for i, project in enumerate(available_projects)}
    
    # Initialize cost matrix with a high value
    # We'll use a value higher than the maximum preference rank
    max_cost = max(len(prefs) for prefs in student_preferences.values()) + 1
    cost_matrix = np.full((num_students, num_projects), max_cost)
    
    # Fill in the cost matrix based on preferences
    # Lower preference (higher index in the list) = higher cost
    for i, (student, preferences) in enumerate(student_preferences.items()):
        for rank, project in enumerate(preferences):
            if project in project_to_index:  # Ensure the project is available
                j = project_to_index[project]
                # Rank + 1 as the cost (1-indexed)
                cost_matrix[i, j] = rank + 1
    
    # Apply the Hungarian algorithm to minimize total cost
    # This will maximize preference satisfaction
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Create the assignment dictionary
    assignments = {}
    students = list(student_preferences.keys())
    
    for i, j in zip(row_indices, col_indices):
        student = students[i]
        project = available_projects[j]
        preference_rank = cost_matrix[i, j]
        
        if preference_rank < max_cost:  # Only assign if it was in the student's preferences
            assignments[student] = {
                'project': project,
                'preference_rank': int(preference_rank)
            }
        else:
            # If no preferred project could be assigned
            assignments[student] = {
                'project': project,
                'preference_rank': None  # Indicates not in preferences
            }
    
    return sort_dict(assignments)

def get_surname(full_name):
    return full_name.split()[-1]  # Assumes the last word is the surname


# Sort by surname
def sort_dict(people):
    sorted_people = dict(sorted(people.items(), key=lambda item: get_surname(item[0])))

    return sorted_people


def print_assignment_summary(assignments):
    """
    Prints a summary of the assignments, showing each student's assigned project
    and how it ranked in their preferences.
    """
    print("\nProject Assignments:")
    print("-" * 50)
    print(f"{'Student':<20} {'Assigned Project':<30} {'Preference Rank':<15}")
    print("-" * 50)
    
    for student, assignment in assignments.items():
        project = assignment['project']
        rank = assignment['preference_rank']
        
        rank_str = f"{rank}" if rank is not None else "Not preferred"
        print(f"{student:<20} {project:<30} {rank_str:<15}")
    
    # Calculate statistics
    ranks = [a['preference_rank'] for a in assignments.values() if a['preference_rank'] is not None]
    if ranks:
        avg_rank = sum(ranks) / len(ranks)
        got_first_choice = ranks.count(1)
        got_top_three = sum(1 for r in ranks if r <= 3)
        
        print("\nAssignment Statistics:")
        print(f"Students who got their first choice: {got_first_choice} ({got_first_choice/len(assignments)*100:.1f}%)")
        print(f"Students who got one of their top three choices: {got_top_three} ({got_top_three/len(assignments)*100:.1f}%)")
        print(f"Average preference rank: {avg_rank:.2f}")
    
    # Check for unassigned students
    unassigned = [student for student, a in assignments.items() if a['preference_rank'] is None]
    if unassigned:
        print(f"\nWarning: {len(unassigned)} students were assigned projects not in their preferences:")
        for student in unassigned:
            print(f"  - {student}: {assignments[student]['project']}")

def save_assignments_to_csv(assignments, output_file):
    """
    Save the project assignments to a CSV file.
    
    Parameters:
    -----------
    assignments : dict
        Dictionary mapping student names to their assigned project and preference rank
    
    output_file : str
        Path to save the output CSV file
    """
    # Prepare data for DataFrame
    data = []
    for student, assignment in assignments.items():
        data.append({
            'Student': student,
            'Assigned Project': assignment['project'],
            'Preference Rank': assignment['preference_rank'] if assignment['preference_rank'] is not None else 'Not preferred'
        })
    
    # Create and save DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"\nAssignments saved to {output_file}")

def save_assignments_to_html(assignments, output_file):
    """
    Save the project assignments to an HTML file with alternating row colors.
    
    Parameters:
    -----------
    assignments : dict
        Dictionary mapping student names to their assigned project and preference rank
    
    output_file : str
        Path to save the output HTML file
    """
    # Prepare data for DataFrame
    data = []
    for student, assignment in assignments.items():
        data.append({
            'Student': student,
            'Assigned Project': assignment['project'],
            'Preference Rank': assignment['preference_rank'] if assignment['preference_rank'] is not None else 'Not preferred'
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate some statistics for the header
    total_students = len(assignments)
    ranks = [a['preference_rank'] for a in assignments.values() if a['preference_rank'] is not None]
    got_first_choice = ranks.count(1) if ranks else 0
    got_top_three = sum(1 for r in ranks if r <= 3) if ranks else 0
    avg_rank = sum(ranks) / len(ranks) if ranks else 0
    
    # Create HTML with styling
    html = f"""
    <head>
        <title>Student Project Assignments</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            h1 {{
                color: #2c3e50;
                margin-bottom: 10px;
            }}
            .stats {{
                margin-bottom: 20px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border-left: 5px solid #2c3e50;
                color: #333;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
            }}
            th {{
                background-color: #2c3e50;
                color: white;
                padding: 10px;
                text-align: left;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #ddd;
            }}
            td {{
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
            .first-choice {{
                background-color: #d4edda;
            }}
            .top-three {{
                background-color: #fff3cd;
            }}
            .not-preferred {{
                background-color: #f8d7da;
            }}
            .footer {{
                margin-top: 20px;
                font-size: 0.8em;
                color: #666;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <h1>Student Project Assignments</h1>
        <div class="stats">
            <p><strong>Total Students:</strong> {total_students}</p>
            <p><strong>Students who got their first choice:</strong> {got_first_choice} ({got_first_choice/total_students*100:.1f}%)</p>
            <p><strong>Students who got one of their top three choices:</strong> {got_top_three} ({got_top_three/total_students*100:.1f}%)</p>
            <p><strong>Average preference rank:</strong> {avg_rank:.2f}</p>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Student</th>
                    <th>Assigned Project</th>
                    <th>Preference Rank</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add rows for each assignment
    for student, assignment in assignments.items():
        project = assignment['project']
        rank = assignment['preference_rank']
        
        # Determine row class based on preference rank
        row_class = ""
        if rank == 1:
            row_class = "first-choice"
        elif rank is not None and rank <= 3:
            row_class = "top-three"
        elif rank is None:
            row_class = "not-preferred"
        
        rank_str = f"{rank}" if rank is not None else "Not preferred"
        
        html += f"""
                <tr class="{row_class}">
                    <td>{student}</td>
                    <td>{project}</td>
                    <td>{rank_str}</td>
                </tr>
        """
    
    # Close the HTML
    html += """
            </tbody>
        </table>
        <div class="footer">
            <p>Generated using Hungarian Algorithm for Optimal Project Assignment</p>
        </div>
    """
    
    # Write the HTML to a file
    with open(output_file, "w") as f:
        f.write(html)
    
    print(f"HTML report saved to {output_file}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Assign projects to students based on preferences.')
    parser.add_argument('input_file', help='Path to the CSV file with student preferences')
    parser.add_argument('--output', '-o', default='project_assignments', help='Base filename for output files (without extension)')
    args = parser.parse_args()
    
    # Parse the CSV file
    print(f"Parsing preferences from {args.input_file}...")
    student_preferences, available_projects = parse_preference_csv(args.input_file)
    
    print(f"Found {len(student_preferences)} students and {len(available_projects)} projects.")
    
    # Run the assignment algorithm
    print("Running assignment algorithm...")
    assignments = assign_projects(student_preferences, available_projects)
    
    # Print the results
    print_assignment_summary(assignments)
    
    # Base filename (without extension) for output files
    base_output = args.output.rsplit('.', 1)[0] if '.' in args.output else args.output
    
    # Save the assignments to a CSV file
    csv_output = f"{base_output}.csv"
    save_assignments_to_csv(assignments, csv_output)
    
    # Save the assignments to an HTML file
    html_output = f"{base_output}.html"
    save_assignments_to_html(assignments, html_output)
