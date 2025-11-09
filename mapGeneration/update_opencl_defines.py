import re
import os

class OpenCLDefineUpdater:
    def __init__(self, cl_file_path):
        self.cl_file_path = cl_file_path
        
    def update_define(self, define_name, value):
        # Read the file
        with open(self.cl_file_path, 'r') as f:
            content = f.read()
        
        # Format the value appropriately
        if isinstance(value, float):
            value_str = f"{value}f"
        else:
            value_str = str(value)
        
        # Pattern to match the define
        pattern = rf'^(#define\s+{define_name}\s+)(.+?)$'
        
        # Replace the define
        new_content = re.sub(pattern, rf'\g<1>{value_str}', content, flags=re.MULTILINE)
        
        # Write back
        with open(self.cl_file_path, 'w') as f:
            f.write(new_content)
        
        print(f"Updated {define_name} = {value_str}")
    
    def update_defines(self, defines_dict):
        # Read the file once
        with open(self.cl_file_path, 'r') as f:
            content = f.read()
        
        # Update each define
        for define_name, value in defines_dict.items():
            # Format the value appropriately
            if isinstance(value, float):
                value_str = f"{value}f"
            else:
                value_str = str(value)
            
            # Pattern to match the define
            pattern = rf'^(#define\s+{define_name}\s+)(.+?)$'
            
            # Replace the define
            content = re.sub(pattern, rf'\g<1>{value_str}', content, flags=re.MULTILINE)
            
            print(f"Updated {define_name} = {value_str}")
        
        # Write back once
        with open(self.cl_file_path, 'w') as f:
            f.write(content)
    
    def get_define(self, define_name):
        with open(self.cl_file_path, 'r') as f:
            content = f.read()
        
        pattern = rf'^#define\s+{define_name}\s+(.+?)$'
        match = re.search(pattern, content, flags=re.MULTILINE)
        
        if match:
            return match.group(1).strip()
        return None