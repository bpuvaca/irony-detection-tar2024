import sys

def find_common_lines(file1_name, file2_name):
    with open(file1_name, 'r', encoding='ISO-8859-1') as file1, open(file2_name, 'r', encoding='ISO-8859-1') as file2:
        lines_file1 = set(file1.readlines())
        lines_file2 = set(file2.readlines())
    
    common_lines = lines_file1.intersection(lines_file2)
    num_common_lines = len(common_lines)
    
    return common_lines, num_common_lines

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide exactly two filenames as arguments.")
        sys.exit(1)
    
    file1_name = sys.argv[1]
    file2_name = sys.argv[2]
    
    common_lines, num_common_lines = find_common_lines(file1_name, file2_name)
    
    print(f"Overlaping tweets ({num_common_lines}):")
    for line in common_lines:
        print(line.strip())



