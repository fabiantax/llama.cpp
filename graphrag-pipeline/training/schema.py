"""Domain schema for GPU optimization knowledge graph NER + RE."""

# 12 entity types (matching main.rs labels)
ENTITY_TYPES = [
    "hardware",
    "gpu_feature",
    "optimization_technique",
    "algorithm",
    "software_framework",
    "performance_metric",
    "memory_pattern",
    "kernel_operation",
    "model_architecture",
    "constraint",
    "data_structure",
    "research_paper",
]

# BIO tags: O + B-<type> + I-<type> for each entity type
BIO_TAGS = ["O"]
for et in ENTITY_TYPES:
    BIO_TAGS.append(f"B-{et}")
    BIO_TAGS.append(f"I-{et}")

BIO_TAG2ID = {tag: i for i, tag in enumerate(BIO_TAGS)}
BIO_ID2TAG = {i: tag for i, tag in enumerate(BIO_TAGS)}

# 20 relation types + no_relation (matching main.rs schema)
RELATION_TYPES = [
    "no_relation",
    "TARGETS",
    "IS_FEATURE_OF",
    "IMPLEMENTS",
    "USES",
    "IMPROVES",
    "REDUCES",
    "MEASURES",
    "LIMITS",
    "IS_PART_OF",
    "BUILDS_ON",
    "EXTENDS",
    "ENABLES",
    "REQUIRES",
    "INTRODUCES",
    "VALIDATES",
    "COMPETES_WITH",
    "OPTIMIZES",
    "ELIMINATES",
    "COULD_IMPROVE",
    "PORTS_TO",
]

REL_TYPE2ID = {rt: i for i, rt in enumerate(RELATION_TYPES)}
REL_ID2TYPE = {i: rt for i, rt in enumerate(RELATION_TYPES)}

# RE entity markers (added as special tokens)
RE_MARKERS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
