Query or modify the FalkorDB gpu_optimization knowledge graph. Pass a Cypher query or describe what you want.

## Instructions

1. Load FalkorDB MCP tools:
   - Use ToolSearch to load: `+falkordb` (finds graph query/create tools)

2. Connection details:
   - Host: localhost:6379 (Redis protocol)
   - Graph name: `gpu_optimization`
   - Browser UI: http://localhost:3000

3. Execute the user's request: $ARGUMENTS

## Common Cypher Patterns

- **List all node labels**: `CALL db.labels()`
- **List all relation types**: `CALL db.relationshipTypes()`
- **Find a node**: `MATCH (n {name: 'SharedMemoryTiling'}) RETURN n`
- **All neighbors**: `MATCH (n {name: 'DeltaNet'})-[r]-(m) RETURN n, type(r), m`
- **Shortest path**: `MATCH p=shortestPath((a {name: 'X'})-[*]-(b {name: 'Y'})) RETURN p`
- **Create node**: `CREATE (:optimization_technique {name: 'MyTech', description: 'desc', type: 'optimization_technique'})`
- **Create relation**: `MATCH (a {name: 'X'}), (b {name: 'Y'}) CREATE (a)-[:IMPROVES]->(b)`
- **Fuzzy search**: `MATCH (n) WHERE n.name CONTAINS 'SSM' RETURN n`

## Relation Types

IMPLEMENTS, USES, OPTIMIZES, TARGETS, IMPROVES, REDUCES, ELIMINATES, MEASURES, LIMITS, ENABLES, EXTENDS, BUILDS_ON, VALIDATES, COMPETES_WITH, IS_PART_OF, IS_FEATURE_OF, REQUIRES, COULD_IMPROVE, INTRODUCES, PORTS_TO

## Entity Types

hardware, gpu_feature, optimization_technique, algorithm, software_framework, performance_metric, memory_pattern, kernel_operation, model_architecture, constraint, data_structure, research_paper
