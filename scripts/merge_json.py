import json
import random
from pathlib import Path

def merge_all_into_one():
    data_dir = Path("data")
    emp_path = data_dir / "employees.json"
    dir_path = data_dir / "directory.json"

    employees = json.loads(emp_path.read_text(encoding="utf-8")) if emp_path.exists() else []
    directory = json.loads(dir_path.read_text(encoding="utf-8")) if dir_path.exists() else []

    # Build a lookup for existing employees by email or name to avoid duplicates
    existing_map = {}
    for emp in employees:
        email = emp.get("email", "").strip().lower()
        first = emp.get("first_name", "")
        last = emp.get("last_name", "")
        name = f"{first} {last}".strip().lower()
        if email:
            existing_map[email] = emp
        elif name:
            existing_map[name] = emp

    merged_list = employees.copy()
    added_count = 0
    dir_id_counter = 1000

    for d in directory:
        # Skip empty alphabet headers
        if not d.get("Role") and not d.get("Department"):
            continue
            
        email = d.get("Email", "").strip()
        full_name = d.get("Name", "").strip()
        name_lower = full_name.lower()
        email_lower = email.lower()

        # If they already exist in employees.json, just update location and phone
        match = existing_map.get(email_lower) or existing_map.get(name_lower)
        if match:
            match["location"] = d.get("Location", "")
            match["phone"] = d.get("Phone", "")
        else:
            # Create a brand new employee record for them using the requested schema
            name_parts = full_name.split(" ", 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ""
            
            new_emp = {
                "employee_id": f"EMP{dir_id_counter}",
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "department": d.get("Department", ""),
                "designation": d.get("Role", ""),
                "joining_date": None,
                "salary": None,
                "project_name": None,
                "project_start_date": None,
                "project_end_date": None,
                "client_name": None,
                "location": d.get("Location", ""),
                "phone": d.get("Phone", "")
            }
            merged_list.append(new_emp)
            added_count += 1
            dir_id_counter += 1

    # Save the combined, unified list back to employees.json
    emp_path.write_text(json.dumps(merged_list, indent=2), encoding="utf-8")
    print(f"Successfully converted and added {added_count} directory records into employees.json")
    print(f"Total records in employees.json: {len(merged_list)}")

if __name__ == "__main__":
    merge_all_into_one()
