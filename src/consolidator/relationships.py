"""Relationship builder — links entities together into a knowledge graph.

Takes raw entities from the ingestors and builds:
  - Employee → Project (works_on)
  - Project → Tool (uses_tech)
  - Policy → Department (applies_to)
  - Employee → Policy (governed_by)
"""

from __future__ import annotations

import logging

from src.consolidator.entities import (
    EmployeeEntity,
    HolidayEntity,
    KnowledgeBase,
    PolicyEntity,
    ProjectEntity,
    Relationship,
    ToolEntity,
)

logger = logging.getLogger(__name__)


def build_knowledge_base(
    employees: list[EmployeeEntity],
    policies: list[PolicyEntity],
    projects: list[ProjectEntity],
    tools: list[ToolEntity],
    holidays: list[HolidayEntity] = None,
) -> KnowledgeBase:
    """Assemble all entities and infer relationships between them."""
    kb = KnowledgeBase()

    # --- Register entities ---
    for emp in employees:
        kb.employees[emp.id] = emp

    for pol in policies:
        kb.policies[pol.id] = pol

    for prj in projects:
        kb.projects[prj.id] = prj

    for tool in tools:
        kb.tools[tool.id] = tool

    if holidays:
        for hol in holidays:
            kb.holidays[hol.id] = hol

    # --- Build relationships ---
    _link_employees_to_projects(kb)
    _link_projects_to_tools(kb)
    _link_employees_to_policies(kb)

    logger.info("Knowledge base built: %s", kb.stats)
    return kb


# ---------------------------------------------------------------------------
# Relationship builders
# ---------------------------------------------------------------------------

def _link_employees_to_projects(kb: KnowledgeBase) -> None:
    """Link employees to their assigned projects."""
    for emp in kb.employees.values():
        if not emp.project_name:
            continue

        # Find or create matching project
        project_id = f"PRJ:{emp.project_name}"

        if project_id not in kb.projects:
            # Create a stub project from employee data
            kb.projects[project_id] = ProjectEntity(
                id=project_id,
                name=emp.project_name,
                about_client=emp.client_name,
            )

        project = kb.projects[project_id]
        if emp.id not in project.employee_ids:
            project.employee_ids.append(emp.id)

        # Update client if the project doesn't have one yet
        if not project.about_client and emp.client_name:
            project.about_client = emp.client_name

        kb.relationships.append(Relationship(
            from_id=emp.id,
            to_id=project_id,
            relation_type="works_on",
            description=f"{emp.name} works on project {emp.project_name}",
        ))

    logger.info(
        "Linked %d employee-project relationships",
        sum(1 for r in kb.relationships if r.relation_type == "works_on"),
    )


def _link_projects_to_tools(kb: KnowledgeBase) -> None:
    """Link projects to the tools/technologies they use."""
    for prj in kb.projects.values():
        for tool_name in prj.tools:
            tool_id = f"TOOL:{tool_name}"
            if tool_id not in kb.tools:
                kb.tools[tool_id] = ToolEntity(id=tool_id, name=tool_name)

            if prj.id not in kb.tools[tool_id].project_ids:
                kb.tools[tool_id].project_ids.append(prj.id)

            kb.relationships.append(Relationship(
                from_id=prj.id,
                to_id=tool_id,
                relation_type="uses_tech",
                description=f"Project {prj.name} uses {tool_name}",
            ))

    logger.info(
        "Linked %d project-tool relationships",
        sum(1 for r in kb.relationships if r.relation_type == "uses_tech"),
    )


def _link_employees_to_policies(kb: KnowledgeBase) -> None:
    """Link employees to applicable policies based on department and scope.

    All employees are linked to general policies. Department-specific logic
    can be added here later.
    """
    general_policies = [
        pol_id for pol_id, pol in kb.policies.items()
        if pol.applicable_to == "All employees"
    ]

    for emp in kb.employees.values():
        for pol_id in general_policies:
            kb.relationships.append(Relationship(
                from_id=emp.id,
                to_id=pol_id,
                relation_type="governed_by",
                description=f"{emp.name} is governed by {kb.policies[pol_id].name}",
            ))

    logger.info(
        "Linked %d employee-policy relationships",
        sum(1 for r in kb.relationships if r.relation_type == "governed_by"),
    )
