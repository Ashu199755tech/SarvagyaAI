"""mock_keka_server.py — Standalone mock Employee API server for ResoAI.

Usage:
    python3 -m uvicorn scripts.mock_keka_server:app --port 8000
"""

from __future__ import annotations

from typing import Any
from fastapi import FastAPI, Query
from pydantic import BaseModel


app = FastAPI(title="Keka Mock HR API", version="1.0.0")

# --- Models ---

class Employee(BaseModel):
    employee_id: str
    first_name: str
    last_name: str
    email: str | None = None
    department: str | None = None
    designation: str | None = None
    joining_date: str | None = None
    salary: float | None = None
    project_name: str | None = None
    project_start_date: str | None = None
    project_end_date: str | None = None
    client_name: str | None = None

class PaginatedResponse(BaseModel):
    total: int
    skip: int
    limit: int
    data: list[Employee]

# --- Static Mock Data ---

EMPLOYEES = [
    {"employee_id": "EMP0001", "first_name": "Anthony", "last_name": "Young", "email": "anthony.young@fiftyfive.tech", "department": "AI/ML", "designation": "AI Architect", "joining_date": "2023-01-15", "salary": 2400000.0, "project_name": "Phoenix", "client_name": "Accenture"},
    {"employee_id": "EMP0002", "first_name": "John", "last_name": "Smith", "email": "john.smith@fiftyfive.tech", "department": "Engineering", "designation": "Staff Engineer", "joining_date": "2022-06-01", "salary": 1800000.0, "project_name": "Atlas", "client_name": "TCS"},
    {"employee_id": "EMP0003", "first_name": "Sarah", "last_name": "Miller", "email": "sarah.miller@fiftyfive.tech", "department": "HR", "designation": "HR Manager", "joining_date": "2021-03-10", "salary": 950000.0, "project_name": None, "client_name": None},
    {"employee_id": "EMP0004", "first_name": "Michael", "last_name": "Brown", "email": "michael.brown@fiftyfive.tech", "department": "Data Science", "designation": "Senior Data Scientist", "joining_date": "2023-04-20", "salary": 1600000.0, "project_name": "Apollo", "client_name": "IBM"},
    {"employee_id": "EMP0005", "first_name": "Emily", "last_name": "Davis", "email": "emily.davis@fiftyfive.tech", "department": "Product", "designation": "Product Manager", "joining_date": "2022-11-05", "salary": 1200000.0, "project_name": "Artemis", "client_name": "Cognizant"},
    {"employee_id": "EMP0006", "first_name": "David", "last_name": "Wilson", "email": "david.wilson@fiftyfive.tech", "department": "DevOps", "designation": "Cloud Engineer", "joining_date": "2023-02-14", "salary": 1500000.0, "project_name": "Hercules", "client_name": "Wipro"},
    {"employee_id": "EMP0007", "first_name": "James", "last_name": "Taylor", "email": "james.taylor@fiftyfive.tech", "department": "Engineering", "designation": "Backend Developer", "joining_date": "2022-09-12", "salary": 1100000.0, "project_name": "Orion", "client_name": "Infosys"},
    {"employee_id": "EMP0008", "first_name": "Linda", "last_name": "Anderson", "email": "linda.anderson@fiftyfive.tech", "department": "Marketing", "designation": "Marketing Lead", "joining_date": "2021-07-22", "salary": 850000.0, "project_name": None, "client_name": None},
    {"employee_id": "EMP0009", "first_name": "Robert", "last_name": "Thomas", "email": "robert.thomas@fiftyfive.tech", "department": "AI/ML", "designation": "Generative AI Lead", "joining_date": "2023-05-30", "salary": 2200000.0, "project_name": "Auggit", "client_name": "Digital Solutions"},
    {"employee_id": "EMP0010", "first_name": "Jennifer", "last_name": "Jackson", "email": "jennifer.jackson@fiftyfive.tech", "department": "Engineering", "designation": "Frontend Developer", "joining_date": "2022-12-01", "salary": 1000000.0, "project_name": "Polaris", "client_name": "Capgemini"},
    {"employee_id": "EMP0011", "first_name": "Anthony", "last_name": "West", "email": "anthony.west@fiftyfive.tech", "department": "AI/ML", "designation": "Senior ML Engineer", "joining_date": "2024-01-10", "salary": 1700000.0, "project_name": "Phoenix", "client_name": "Accenture"},
    {"employee_id": "EMP0012", "first_name": "Charles", "last_name": "Harris", "email": "charles.harris@fiftyfive.tech", "department": "Data Science", "designation": "Data Scientist", "joining_date": "2023-08-15", "salary": 1300000.0, "project_name": "Apollo", "client_name": "IBM"},
    {"employee_id": "EMP0013", "first_name": "Patricia", "last_name": "Martin", "email": "patricia.martin@fiftyfive.tech", "department": "Engineering", "designation": "QA Engineer", "joining_date": "2022-05-20", "salary": 900000.0, "project_name": "Atlas", "client_name": "TCS"},
    {"employee_id": "EMP0014", "first_name": "Christopher", "last_name": "Lee", "email": "christopher.lee@fiftyfive.tech", "department": "Product", "designation": "Associate Product Manager", "joining_date": "2023-10-01", "salary": 800000.0, "project_name": "Artemis", "client_name": "Cognizant"},
    {"employee_id": "EMP0015", "first_name": "Barbara", "last_name": "Walker", "email": "barbara.walker@fiftyfive.tech", "department": "HR", "designation": "HR Generalist", "joining_date": "2021-11-15", "salary": 700000.0, "project_name": None, "client_name": None},
    {"employee_id": "EMP0016", "first_name": "Daniel", "last_name": "Allen", "email": "daniel.allen@fiftyfive.tech", "department": "DevOps", "designation": "SRE", "joining_date": "2023-03-25", "salary": 1400000.0, "project_name": "Hercules", "client_name": "Wipro"},
    {"employee_id": "EMP0017", "first_name": "Matthew", "last_name": "King", "email": "matthew.king@fiftyfive.tech", "department": "Engineering", "designation": "Software Developer", "joining_date": "2022-07-01", "salary": 950000.0, "project_name": "Orion", "client_name": "Infosys"},
    {"employee_id": "EMP0018", "first_name": "Susan", "last_name": "Wright", "email": "susan.wright@fiftyfive.tech", "department": "Marketing", "designation": "Content Specialist", "joining_date": "2021-09-10", "salary": 650000.0, "project_name": None, "client_name": None},
    {"employee_id": "EMP0019", "first_name": "Joseph", "last_name": "Scott", "email": "joseph.scott@fiftyfive.tech", "department": "AI/ML", "designation": "ML Researcher", "joining_date": "2023-06-15", "salary": 2000000.0, "project_name": "Auggit", "client_name": "Digital Solutions"},
    {"employee_id": "EMP0020", "first_name": "Margaret", "last_name": "Green", "email": "margaret.green@fiftyfive.tech", "department": "Engineering", "designation": "UI Developer", "joining_date": "2022-10-30", "salary": 850000.0, "project_name": "Polaris", "client_name": "Capgemini"},
    {"employee_id": "EMP0021", "first_name": "Richard", "last_name": "Baker", "email": "richard.baker@fiftyfive.tech", "department": "Sales", "designation": "Sales Manager", "joining_date": "2021-05-05", "salary": 1100000.0, "project_name": None, "client_name": None},
    {"employee_id": "EMP0022", "first_name": "Thomas", "last_name": "Nelson", "email": "thomas.nelson@fiftyfive.tech", "department": "Engineering", "designation": "Principal Engineer", "joining_date": "2020-01-01", "salary": 2800000.0, "project_name": "Zeus", "client_name": "FiftyFive"},
    {"employee_id": "EMP0023", "first_name": "Karen", "last_name": "Carter", "email": "karen.carter@fiftyfive.tech", "department": "Data Science", "designation": "Data Lead", "joining_date": "2022-02-15", "salary": 1900000.0, "project_name": "Ares", "client_name": "Amazon"},
    {"employee_id": "EMP0024", "first_name": "Kevin", "last_name": "Mitchell", "email": "kevin.mitchell@fiftyfive.tech", "department": "Engineering", "designation": "Fullstack Developer", "joining_date": "2023-07-20", "salary": 1300000.0, "project_name": "Zeus", "client_name": "FiftyFive"},
    {"employee_id": "EMP0025", "first_name": "Sandra", "last_name": "Perez", "email": "sandra.perez@fiftyfive.tech", "department": "AI/ML", "designation": "Software Engineer (AI)", "joining_date": "2023-09-01", "salary": 1400000.0, "project_name": "Ares", "client_name": "Amazon"},
]

# --- Endpoints ---

@app.get("/employees", response_model=PaginatedResponse)
async def get_employees(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """Returns a list of employee records with pagination."""
    data = EMPLOYEES[skip : skip + limit]
    return PaginatedResponse(
        total=len(EMPLOYEES),
        skip=skip,
        limit=limit,
        data=data,
    )

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Employee Mock API is alive"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
