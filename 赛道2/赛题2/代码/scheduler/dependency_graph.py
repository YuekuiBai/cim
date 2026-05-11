"""
Dependency Graph for scheduling
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set
from collections import defaultdict


@dataclass
class DependencyGraph:
    """Represents operator dependencies"""
    dependencies: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    dependents: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    
    def add_dependency(self, operator: str, depends_on: str):
        """Add a dependency: operator depends on depends_on"""
        self.dependencies[operator].append(depends_on)
        self.dependents[depends_on].append(operator)
    
    def get_dependencies(self, operator: str) -> List[str]:
        """Get all dependencies of an operator"""
        return self.dependencies.get(operator, [])
    
    def get_dependents(self, operator: str) -> List[str]:
        """Get all operators that depend on this operator"""
        return self.dependents.get(operator, [])
    
    def get_ready_operators(self, completed: Set[str]) -> List[str]:
        """Get operators that are ready to execute (all deps completed)"""
        ready = []
        for operator in self.dependencies:
            if operator in completed:
                continue
            deps = self.dependencies[operator]
            if all(dep in completed for dep in deps):
                ready.append(operator)
        return ready
