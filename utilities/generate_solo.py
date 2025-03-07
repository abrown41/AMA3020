import csv
import re

def extract_project_titles_from_fulllist(fulllist_path):
    """
    Extract project titles from the full_list.tex file.
    
    Returns a dictionary mapping project input commands to their titles.
    """
    project_map = {}
    
    try:
        with open(fulllist_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Pattern to match \input{projectX} %: project title format
            pattern = r'\\input\{([^}]+)\}\s*%:\s*(.+)$'
            matches = re.findall(pattern, content, re.MULTILINE)
            
            for project_file, title in matches:
                project_map[title.strip()] = project_file.strip()
                
        return project_map
    except Exception as e:
        print(f"Error reading full_list.tex file: {e}")
        return {}

def get_chosen_projects(csv_path):
    """
    Read the CSV file and return a set of chosen project titles.
    """
    chosen_projects = set()
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Assuming 'Assigned Project' is the exact column name
                project_title = row.get('Assigned Project', '').strip()
                
                if project_title:
                    chosen_projects.add(project_title)
                    
        return chosen_projects
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return set()

def create_partial_list(fulllist_path, output_path, project_map, chosen_projects):
    """
    Create the partial_list.tex file containing only unchosen projects.
    """
    try:
        with open(fulllist_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            
            with open(output_path, 'w', encoding='utf-8') as outfile:
                outfile.write("% This file was automatically generated\n")
                outfile.write("% It contains only the projects that have not been chosen\n\n")
                
                for line in lines:
                    if line.strip():
                        match = re.match(r'\\input\{([^}]+)\}\s*%:\s*(.+)$', line)
                        
                        if match:
                            project_file = match.group(1).strip()
                            title = match.group(2).strip()
                            
                            if title not in chosen_projects:
                                outfile.write(line)
                
        print(f"Successfully created {output_path}")
    except Exception as e:
        print(f"Error creating partial list: {e}")

def main():
    # File paths - Change these to match your actual file locations
    from argparse import ArgumentParser as AP 
    parser = AP()
    parser.add_argument("-f", '--file', 
                        help="path to csv file containing chosen projects",
                        default="project_assignments.csv")
    parser.add_argument("-l", "--latex_file",
                        help="if provided, run pdflatex on the latex file",
                        default=None)


    args = parser.parse_args()

    csv_path = args.file
    fulllist_path = "fulllist.tex"
    output_path = "partial_list.tex"
    
    # Get chosen projects from CSV
    chosen_projects = get_chosen_projects(csv_path)
    print(f"Found {len(chosen_projects)} chosen projects")
    
    # Extract project titles and their corresponding files
    project_map = extract_project_titles_from_fulllist(fulllist_path)
    print(f"Found {len(project_map)} projects in the full list")
    
    # Create the partial list
    create_partial_list(fulllist_path, output_path, project_map, chosen_projects)

    if args.latex_file:
        from subprocess import run
        run(["pdflatex", args.latex_file])


if __name__ == "__main__":
    main()
