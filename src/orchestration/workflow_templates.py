"""
Workflow Templates System
Pre-built workflow patterns for common multi-agent scenarios
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
from pathlib import Path

from pydantic import BaseModel, Field, validator

from .workflow_engine import WorkflowDefinition, WorkflowStep, WorkflowCondition, StepType, ConditionType

logger = logging.getLogger(__name__)


class TemplateCategory(str, Enum):
    """Categories for workflow templates"""
    COLLABORATION = "collaboration"
    DECISION_MAKING = "decision_making"
    RESEARCH = "research"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    COORDINATION = "coordination"
    INTEGRATION = "integration"
    AUTOMATION = "automation"


class TemplateDifficulty(str, Enum):
    """Template complexity levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TemplateParameter(BaseModel):
    """Template parameter definition"""
    name: str
    description: str
    type: str  # "string", "number", "boolean", "array", "object"
    required: bool = True
    default_value: Optional[Any] = None
    validation_rules: Optional[Dict[str, Any]] = None

    class Config:
        extra = "forbid"


class TemplateCustomization(BaseModel):
    """Template customization options"""
    allow_step_modification: bool = True
    allow_condition_modification: bool = True
    allow_parameter_override: bool = True
    allowed_roles: Optional[List[str]] = None
    required_capabilities: Optional[List[str]] = None
    min_agents: int = 1
    max_agents: Optional[int] = None

    class Config:
        extra = "forbid"


class WorkflowTemplate(BaseModel):
    """Template for creating workflows"""
    template_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    category: TemplateCategory
    difficulty: TemplateDifficulty
    version: str = "1.0.0"

    # Template structure
    parameters: List[TemplateParameter] = Field(default_factory=list)
    workflow_definition: WorkflowDefinition
    customization: TemplateCustomization = Field(default_factory=TemplateCustomization)

    # Metadata
    tags: List[str] = Field(default_factory=list)
    use_cases: List[str] = Field(default_factory=list)
    estimated_duration_minutes: Optional[int] = None
    success_criteria: List[str] = Field(default_factory=list)

    # Authoring
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    usage_count: int = 0

    class Config:
        extra = "forbid"

    @validator('parameters')
    def validate_parameters(cls, v):
        """Validate parameter names are unique"""
        names = [p.name for p in v]
        if len(names) != len(set(names)):
            raise ValueError("Parameter names must be unique")
        return v


class TemplateLibrary(BaseModel):
    """Collection of workflow templates"""
    library_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = "Default Template Library"
    description: str = "Standard workflow templates"
    version: str = "1.0.0"

    templates: Dict[str, WorkflowTemplate] = Field(default_factory=dict)
    categories: Dict[TemplateCategory, List[str]] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "forbid"

    def add_template(self, template: WorkflowTemplate) -> None:
        """Add a template to the library"""
        self.templates[template.template_id] = template

        # Update category index
        if template.category not in self.categories:
            self.categories[template.category] = []
        if template.template_id not in self.categories[template.category]:
            self.categories[template.category].append(template.template_id)

        self.updated_at = datetime.utcnow()
        logger.info(f"Added template {template.name} to library")

    def remove_template(self, template_id: str) -> bool:
        """Remove a template from the library"""
        if template_id not in self.templates:
            return False

        template = self.templates[template_id]

        # Remove from category index
        if template.category in self.categories:
            if template_id in self.categories[template.category]:
                self.categories[template.category].remove(template_id)

        del self.templates[template_id]
        self.updated_at = datetime.utcnow()
        logger.info(f"Removed template {template.name} from library")
        return True

    def get_templates_by_category(self, category: TemplateCategory) -> List[WorkflowTemplate]:
        """Get all templates in a category"""
        template_ids = self.categories.get(category, [])
        return [self.templates[tid] for tid in template_ids if tid in self.templates]

    def search_templates(self,
                        query: Optional[str] = None,
                        category: Optional[TemplateCategory] = None,
                        difficulty: Optional[TemplateDifficulty] = None,
                        tags: Optional[List[str]] = None) -> List[WorkflowTemplate]:
        """Search templates by criteria"""
        results = list(self.templates.values())

        if category:
            results = [t for t in results if t.category == category]

        if difficulty:
            results = [t for t in results if t.difficulty == difficulty]

        if tags:
            results = [t for t in results if any(tag in t.tags for tag in tags)]

        if query:
            query_lower = query.lower()
            results = [t for t in results if
                      query_lower in t.name.lower() or
                      query_lower in t.description.lower() or
                      any(query_lower in tag.lower() for tag in t.tags)]

        return sorted(results, key=lambda x: x.usage_count, reverse=True)


class WorkflowTemplateManager:
    """Manager for workflow templates"""

    def __init__(self, library_path: Optional[str] = None):
        self.library_path = library_path
        self.library = TemplateLibrary()
        self._initialize_default_templates()

        if library_path:
            self.load_library(library_path)

    def _initialize_default_templates(self) -> None:
        """Initialize with default templates"""
        # Collaboration templates
        self._create_brainstorming_template()
        self._create_peer_review_template()
        self._create_consensus_building_template()

        # Decision making templates
        self._create_pros_cons_analysis_template()
        self._create_multi_criteria_decision_template()
        self._create_risk_assessment_template()

        # Research templates
        self._create_literature_review_template()
        self._create_fact_checking_template()
        self._create_competitive_analysis_template()

        # Problem solving templates
        self._create_root_cause_analysis_template()
        self._create_solution_design_template()
        self._create_implementation_planning_template()

        logger.info("Initialized default workflow templates")

    def _create_brainstorming_template(self) -> None:
        """Create brainstorming workflow template"""
        steps = [
            WorkflowStep(
                step_id="ideation_phase",
                name="Idea Generation",
                type=StepType.PARALLEL_TASKS,
                agent_roles=["creative", "innovator", "analyst"],
                instructions="Generate diverse ideas for the given topic",
                parameters={"topic": "{{topic}}", "duration_minutes": 10},
                parallel_tasks=[
                    {"agent_role": "creative", "task": "Generate creative solutions"},
                    {"agent_role": "innovator", "task": "Think of innovative approaches"},
                    {"agent_role": "analyst", "task": "Analyze problem systematically"}
                ]
            ),
            WorkflowStep(
                step_id="consolidation",
                name="Idea Consolidation",
                type=StepType.AGENT_TASK,
                agent_roles=["facilitator"],
                instructions="Consolidate and categorize all generated ideas",
                parameters={"previous_results": "{{ideation_phase.results}}"}
            ),
            WorkflowStep(
                step_id="evaluation",
                name="Idea Evaluation",
                type=StepType.AGENT_TASK,
                agent_roles=["critic", "analyst"],
                instructions="Evaluate ideas for feasibility and impact",
                parameters={"ideas": "{{consolidation.results}}"}
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="brainstorming_template",
            name="Collaborative Brainstorming",
            description="Multi-agent brainstorming session",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Collaborative Brainstorming",
            description="Structured brainstorming session with idea generation, consolidation, and evaluation",
            category=TemplateCategory.COLLABORATION,
            difficulty=TemplateDifficulty.BASIC,
            parameters=[
                TemplateParameter(
                    name="topic",
                    description="The topic or problem to brainstorm about",
                    type="string",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["brainstorming", "collaboration", "ideation"],
            use_cases=["Product development", "Problem solving", "Strategic planning"],
            estimated_duration_minutes=30,
            success_criteria=["Generate at least 10 ideas", "Categorize ideas by theme", "Evaluate top 5 ideas"]
        )

        self.library.add_template(template)

    def _create_peer_review_template(self) -> None:
        """Create peer review workflow template"""
        steps = [
            WorkflowStep(
                step_id="initial_review",
                name="Initial Review",
                type=StepType.PARALLEL_TASKS,
                agent_roles=["critic", "specialist"],
                instructions="Conduct thorough review of the work",
                parameters={"work_to_review": "{{work_content}}"},
                parallel_tasks=[
                    {"agent_role": "critic", "task": "Critical analysis and feedback"},
                    {"agent_role": "specialist", "task": "Technical accuracy review"}
                ]
            ),
            WorkflowStep(
                step_id="feedback_consolidation",
                name="Feedback Consolidation",
                type=StepType.AGENT_TASK,
                agent_roles=["moderator"],
                instructions="Consolidate feedback from reviewers",
                parameters={"reviews": "{{initial_review.results}}"}
            ),
            WorkflowStep(
                step_id="improvement_suggestions",
                name="Improvement Suggestions",
                type=StepType.AGENT_TASK,
                agent_roles=["advisor"],
                instructions="Provide constructive improvement suggestions",
                parameters={"consolidated_feedback": "{{feedback_consolidation.results}}"}
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="peer_review_template",
            name="Peer Review Process",
            description="Structured peer review workflow",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Peer Review Process",
            description="Comprehensive peer review with multiple perspectives and constructive feedback",
            category=TemplateCategory.VALIDATION,
            difficulty=TemplateDifficulty.INTERMEDIATE,
            parameters=[
                TemplateParameter(
                    name="work_content",
                    description="The work content to be reviewed",
                    type="string",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["peer-review", "validation", "feedback"],
            use_cases=["Code review", "Document review", "Research validation"],
            estimated_duration_minutes=45
        )

        self.library.add_template(template)

    def _create_consensus_building_template(self) -> None:
        """Create consensus building workflow template"""
        steps = [
            WorkflowStep(
                step_id="position_gathering",
                name="Position Gathering",
                type=StepType.PARALLEL_TASKS,
                agent_roles=["analyst", "facilitator", "advisor"],
                instructions="Gather different positions on the topic",
                parameters={"topic": "{{discussion_topic}}"},
                parallel_tasks=[
                    {"agent_role": "analyst", "task": "Analyze different viewpoints"},
                    {"agent_role": "facilitator", "task": "Identify common ground"},
                    {"agent_role": "advisor", "task": "Suggest compromise solutions"}
                ]
            ),
            WorkflowStep(
                step_id="conflict_identification",
                name="Conflict Identification",
                type=StepType.AGENT_TASK,
                agent_roles=["moderator"],
                instructions="Identify areas of disagreement",
                parameters={"positions": "{{position_gathering.results}}"}
            ),
            WorkflowStep(
                step_id="consensus_building",
                name="Consensus Building",
                type=StepType.SEQUENTIAL_TASKS,
                agent_roles=["facilitator", "moderator"],
                instructions="Build consensus through structured discussion",
                parameters={"conflicts": "{{conflict_identification.results}}"},
                sequential_tasks=[
                    {"agent_role": "facilitator", "task": "Facilitate discussion"},
                    {"agent_role": "moderator", "task": "Moderate and guide to consensus"}
                ]
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="consensus_building_template",
            name="Consensus Building",
            description="Structured consensus building process",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Consensus Building",
            description="Structured process to build consensus among multiple viewpoints",
            category=TemplateCategory.DECISION_MAKING,
            difficulty=TemplateDifficulty.ADVANCED,
            parameters=[
                TemplateParameter(
                    name="discussion_topic",
                    description="The topic requiring consensus",
                    type="string",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["consensus", "decision-making", "collaboration"],
            use_cases=["Team decisions", "Strategy alignment", "Conflict resolution"],
            estimated_duration_minutes=60
        )

        self.library.add_template(template)

    def _create_pros_cons_analysis_template(self) -> None:
        """Create pros and cons analysis template"""
        steps = [
            WorkflowStep(
                step_id="pros_analysis",
                name="Pros Analysis",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Identify and analyze the advantages",
                parameters={"decision_topic": "{{decision_topic}}"}
            ),
            WorkflowStep(
                step_id="cons_analysis",
                name="Cons Analysis",
                type=StepType.AGENT_TASK,
                agent_roles=["critic"],
                instructions="Identify and analyze the disadvantages",
                parameters={"decision_topic": "{{decision_topic}}"}
            ),
            WorkflowStep(
                step_id="impact_assessment",
                name="Impact Assessment",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Assess the impact of each pro and con",
                parameters={
                    "pros": "{{pros_analysis.results}}",
                    "cons": "{{cons_analysis.results}}"
                }
            ),
            WorkflowStep(
                step_id="recommendation",
                name="Final Recommendation",
                type=StepType.AGENT_TASK,
                agent_roles=["advisor"],
                instructions="Provide final recommendation based on analysis",
                parameters={"impact_assessment": "{{impact_assessment.results}}"}
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="pros_cons_template",
            name="Pros and Cons Analysis",
            description="Structured pros and cons decision analysis",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Pros and Cons Analysis",
            description="Systematic analysis of advantages and disadvantages for decision making",
            category=TemplateCategory.DECISION_MAKING,
            difficulty=TemplateDifficulty.BASIC,
            parameters=[
                TemplateParameter(
                    name="decision_topic",
                    description="The decision or option to analyze",
                    type="string",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["decision-making", "analysis", "evaluation"],
            use_cases=["Business decisions", "Personal choices", "Strategic planning"],
            estimated_duration_minutes=25
        )

        self.library.add_template(template)

    def _create_multi_criteria_decision_template(self) -> None:
        """Create multi-criteria decision analysis template"""
        steps = [
            WorkflowStep(
                step_id="criteria_definition",
                name="Define Criteria",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Define and weight decision criteria",
                parameters={"decision_context": "{{decision_context}}"}
            ),
            WorkflowStep(
                step_id="option_evaluation",
                name="Evaluate Options",
                type=StepType.PARALLEL_TASKS,
                agent_roles=["analyst", "specialist"],
                instructions="Evaluate each option against criteria",
                parameters={
                    "options": "{{options}}",
                    "criteria": "{{criteria_definition.results}}"
                },
                parallel_tasks=[
                    {"agent_role": "analyst", "task": "Quantitative evaluation"},
                    {"agent_role": "specialist", "task": "Qualitative assessment"}
                ]
            ),
            WorkflowStep(
                step_id="scoring_calculation",
                name="Calculate Scores",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Calculate weighted scores for each option",
                parameters={"evaluations": "{{option_evaluation.results}}"}
            ),
            WorkflowStep(
                step_id="sensitivity_analysis",
                name="Sensitivity Analysis",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Perform sensitivity analysis on weights",
                parameters={"scores": "{{scoring_calculation.results}}"}
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="multi_criteria_decision_template",
            name="Multi-Criteria Decision Analysis",
            description="Comprehensive multi-criteria decision analysis",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Multi-Criteria Decision Analysis",
            description="Systematic evaluation of options using multiple weighted criteria",
            category=TemplateCategory.DECISION_MAKING,
            difficulty=TemplateDifficulty.ADVANCED,
            parameters=[
                TemplateParameter(
                    name="decision_context",
                    description="Context and background of the decision",
                    type="string",
                    required=True
                ),
                TemplateParameter(
                    name="options",
                    description="List of options to evaluate",
                    type="array",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["decision-making", "multi-criteria", "analysis"],
            use_cases=["Vendor selection", "Investment decisions", "Strategic choices"],
            estimated_duration_minutes=90
        )

        self.library.add_template(template)

    def _create_risk_assessment_template(self) -> None:
        """Create risk assessment workflow template"""
        steps = [
            WorkflowStep(
                step_id="risk_identification",
                name="Risk Identification",
                type=StepType.PARALLEL_TASKS,
                agent_roles=["analyst", "specialist", "critic"],
                instructions="Identify potential risks",
                parameters={"context": "{{risk_context}}"},
                parallel_tasks=[
                    {"agent_role": "analyst", "task": "Systematic risk analysis"},
                    {"agent_role": "specialist", "task": "Domain-specific risks"},
                    {"agent_role": "critic", "task": "Worst-case scenarios"}
                ]
            ),
            WorkflowStep(
                step_id="probability_assessment",
                name="Probability Assessment",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Assess probability of each risk",
                parameters={"risks": "{{risk_identification.results}}"}
            ),
            WorkflowStep(
                step_id="impact_assessment",
                name="Impact Assessment",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Assess potential impact of each risk",
                parameters={"risks": "{{risk_identification.results}}"}
            ),
            WorkflowStep(
                step_id="mitigation_strategies",
                name="Mitigation Strategies",
                type=StepType.AGENT_TASK,
                agent_roles=["advisor"],
                instructions="Develop risk mitigation strategies",
                parameters={
                    "probabilities": "{{probability_assessment.results}}",
                    "impacts": "{{impact_assessment.results}}"
                }
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="risk_assessment_template",
            name="Risk Assessment",
            description="Comprehensive risk assessment workflow",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Risk Assessment",
            description="Systematic identification, assessment, and mitigation of risks",
            category=TemplateCategory.ANALYSIS,
            difficulty=TemplateDifficulty.INTERMEDIATE,
            parameters=[
                TemplateParameter(
                    name="risk_context",
                    description="Context for risk assessment (project, decision, etc.)",
                    type="string",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["risk", "assessment", "analysis", "mitigation"],
            use_cases=["Project planning", "Investment decisions", "Strategic planning"],
            estimated_duration_minutes=60
        )

        self.library.add_template(template)

    def _create_literature_review_template(self) -> None:
        """Create literature review workflow template"""
        steps = [
            WorkflowStep(
                step_id="source_gathering",
                name="Source Gathering",
                type=StepType.PARALLEL_TASKS,
                agent_roles=["researcher", "analyst"],
                instructions="Gather relevant sources and literature",
                parameters={"research_topic": "{{research_topic}}"},
                parallel_tasks=[
                    {"agent_role": "researcher", "task": "Find academic sources"},
                    {"agent_role": "analyst", "task": "Find industry reports"}
                ]
            ),
            WorkflowStep(
                step_id="source_evaluation",
                name="Source Evaluation",
                type=StepType.AGENT_TASK,
                agent_roles=["critic"],
                instructions="Evaluate source quality and relevance",
                parameters={"sources": "{{source_gathering.results}}"}
            ),
            WorkflowStep(
                step_id="content_analysis",
                name="Content Analysis",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Analyze and synthesize content",
                parameters={"validated_sources": "{{source_evaluation.results}}"}
            ),
            WorkflowStep(
                step_id="gap_identification",
                name="Gap Identification",
                type=StepType.AGENT_TASK,
                agent_roles=["researcher"],
                instructions="Identify gaps in current literature",
                parameters={"analysis": "{{content_analysis.results}}"}
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="literature_review_template",
            name="Literature Review",
            description="Systematic literature review process",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Literature Review",
            description="Comprehensive literature review with source evaluation and gap analysis",
            category=TemplateCategory.RESEARCH,
            difficulty=TemplateDifficulty.INTERMEDIATE,
            parameters=[
                TemplateParameter(
                    name="research_topic",
                    description="The research topic for literature review",
                    type="string",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["research", "literature", "analysis"],
            use_cases=["Academic research", "Market research", "Technology assessment"],
            estimated_duration_minutes=120
        )

        self.library.add_template(template)

    def _create_fact_checking_template(self) -> None:
        """Create fact checking workflow template"""
        steps = [
            WorkflowStep(
                step_id="claim_extraction",
                name="Claim Extraction",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Extract verifiable claims from content",
                parameters={"content": "{{content_to_check}}"}
            ),
            WorkflowStep(
                step_id="source_verification",
                name="Source Verification",
                type=StepType.PARALLEL_TASKS,
                agent_roles=["researcher", "specialist"],
                instructions="Verify claims against reliable sources",
                parameters={"claims": "{{claim_extraction.results}}"},
                parallel_tasks=[
                    {"agent_role": "researcher", "task": "Find authoritative sources"},
                    {"agent_role": "specialist", "task": "Cross-reference multiple sources"}
                ]
            ),
            WorkflowStep(
                step_id="accuracy_assessment",
                name="Accuracy Assessment",
                type=StepType.AGENT_TASK,
                agent_roles=["critic"],
                instructions="Assess accuracy of each claim",
                parameters={"verifications": "{{source_verification.results}}"}
            ),
            WorkflowStep(
                step_id="report_generation",
                name="Report Generation",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Generate fact-checking report",
                parameters={"assessments": "{{accuracy_assessment.results}}"}
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="fact_checking_template",
            name="Fact Checking",
            description="Systematic fact checking process",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Fact Checking",
            description="Systematic verification of claims and statements",
            category=TemplateCategory.VALIDATION,
            difficulty=TemplateDifficulty.INTERMEDIATE,
            parameters=[
                TemplateParameter(
                    name="content_to_check",
                    description="Content that needs fact checking",
                    type="string",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["fact-checking", "verification", "validation"],
            use_cases=["News verification", "Research validation", "Content review"],
            estimated_duration_minutes=45
        )

        self.library.add_template(template)

    def _create_competitive_analysis_template(self) -> None:
        """Create competitive analysis workflow template"""
        steps = [
            WorkflowStep(
                step_id="competitor_identification",
                name="Competitor Identification",
                type=StepType.AGENT_TASK,
                agent_roles=["researcher"],
                instructions="Identify key competitors in the market",
                parameters={"market_context": "{{market_context}}"}
            ),
            WorkflowStep(
                step_id="competitor_analysis",
                name="Competitor Analysis",
                type=StepType.PARALLEL_TASKS,
                agent_roles=["analyst", "specialist"],
                instructions="Analyze competitor strengths and weaknesses",
                parameters={"competitors": "{{competitor_identification.results}}"},
                parallel_tasks=[
                    {"agent_role": "analyst", "task": "Financial and market analysis"},
                    {"agent_role": "specialist", "task": "Product and service analysis"}
                ]
            ),
            WorkflowStep(
                step_id="positioning_analysis",
                name="Positioning Analysis",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Analyze competitive positioning",
                parameters={"competitor_data": "{{competitor_analysis.results}}"}
            ),
            WorkflowStep(
                step_id="opportunity_identification",
                name="Opportunity Identification",
                type=StepType.AGENT_TASK,
                agent_roles=["advisor"],
                instructions="Identify market opportunities and gaps",
                parameters={"positioning": "{{positioning_analysis.results}}"}
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="competitive_analysis_template",
            name="Competitive Analysis",
            description="Comprehensive competitive analysis",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Competitive Analysis",
            description="Systematic analysis of competitors and market positioning",
            category=TemplateCategory.ANALYSIS,
            difficulty=TemplateDifficulty.ADVANCED,
            parameters=[
                TemplateParameter(
                    name="market_context",
                    description="Market or industry context for analysis",
                    type="string",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["competitive", "analysis", "market", "strategy"],
            use_cases=["Market entry", "Strategic planning", "Product development"],
            estimated_duration_minutes=90
        )

        self.library.add_template(template)

    def _create_root_cause_analysis_template(self) -> None:
        """Create root cause analysis workflow template"""
        steps = [
            WorkflowStep(
                step_id="problem_definition",
                name="Problem Definition",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Clearly define the problem",
                parameters={"problem_description": "{{problem_description}}"}
            ),
            WorkflowStep(
                step_id="data_collection",
                name="Data Collection",
                type=StepType.PARALLEL_TASKS,
                agent_roles=["researcher", "specialist"],
                instructions="Collect relevant data and evidence",
                parameters={"problem": "{{problem_definition.results}}"},
                parallel_tasks=[
                    {"agent_role": "researcher", "task": "Historical data analysis"},
                    {"agent_role": "specialist", "task": "Technical investigation"}
                ]
            ),
            WorkflowStep(
                step_id="cause_analysis",
                name="Cause Analysis",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Analyze potential causes using 5 Whys method",
                parameters={"data": "{{data_collection.results}}"}
            ),
            WorkflowStep(
                step_id="root_cause_identification",
                name="Root Cause Identification",
                type=StepType.AGENT_TASK,
                agent_roles=["specialist"],
                instructions="Identify the root cause(s)",
                parameters={"cause_analysis": "{{cause_analysis.results}}"}
            ),
            WorkflowStep(
                step_id="solution_recommendations",
                name="Solution Recommendations",
                type=StepType.AGENT_TASK,
                agent_roles=["advisor"],
                instructions="Recommend solutions to address root causes",
                parameters={"root_causes": "{{root_cause_identification.results}}"}
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="root_cause_analysis_template",
            name="Root Cause Analysis",
            description="Systematic root cause analysis",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Root Cause Analysis",
            description="Systematic investigation to identify underlying causes of problems",
            category=TemplateCategory.PROBLEM_SOLVING,
            difficulty=TemplateDifficulty.INTERMEDIATE,
            parameters=[
                TemplateParameter(
                    name="problem_description",
                    description="Description of the problem to analyze",
                    type="string",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["problem-solving", "analysis", "root-cause"],
            use_cases=["Quality issues", "System failures", "Process problems"],
            estimated_duration_minutes=75
        )

        self.library.add_template(template)

    def _create_solution_design_template(self) -> None:
        """Create solution design workflow template"""
        steps = [
            WorkflowStep(
                step_id="requirements_gathering",
                name="Requirements Gathering",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Gather and analyze solution requirements",
                parameters={"problem_context": "{{problem_context}}"}
            ),
            WorkflowStep(
                step_id="solution_brainstorming",
                name="Solution Brainstorming",
                type=StepType.PARALLEL_TASKS,
                agent_roles=["creative", "innovator", "specialist"],
                instructions="Generate potential solution approaches",
                parameters={"requirements": "{{requirements_gathering.results}}"},
                parallel_tasks=[
                    {"agent_role": "creative", "task": "Creative solution approaches"},
                    {"agent_role": "innovator", "task": "Innovative technologies"},
                    {"agent_role": "specialist", "task": "Technical feasibility"}
                ]
            ),
            WorkflowStep(
                step_id="solution_evaluation",
                name="Solution Evaluation",
                type=StepType.AGENT_TASK,
                agent_roles=["critic"],
                instructions="Evaluate solutions against criteria",
                parameters={"solutions": "{{solution_brainstorming.results}}"}
            ),
            WorkflowStep(
                step_id="design_specification",
                name="Design Specification",
                type=StepType.AGENT_TASK,
                agent_roles=["specialist"],
                instructions="Create detailed design specification",
                parameters={"selected_solution": "{{solution_evaluation.results}}"}
            ),
            WorkflowStep(
                step_id="validation_planning",
                name="Validation Planning",
                type=StepType.AGENT_TASK,
                agent_roles=["validator"],
                instructions="Plan solution validation approach",
                parameters={"design": "{{design_specification.results}}"}
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="solution_design_template",
            name="Solution Design",
            description="Comprehensive solution design process",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Solution Design",
            description="Systematic approach to designing solutions from requirements to validation",
            category=TemplateCategory.PROBLEM_SOLVING,
            difficulty=TemplateDifficulty.ADVANCED,
            parameters=[
                TemplateParameter(
                    name="problem_context",
                    description="Context and description of the problem to solve",
                    type="string",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["solution-design", "requirements", "validation"],
            use_cases=["Product development", "System design", "Process improvement"],
            estimated_duration_minutes=120
        )

        self.library.add_template(template)

    def _create_implementation_planning_template(self) -> None:
        """Create implementation planning workflow template"""
        steps = [
            WorkflowStep(
                step_id="task_breakdown",
                name="Task Breakdown",
                type=StepType.AGENT_TASK,
                agent_roles=["coordinator"],
                instructions="Break down implementation into tasks",
                parameters={"solution_design": "{{solution_design}}"}
            ),
            WorkflowStep(
                step_id="resource_planning",
                name="Resource Planning",
                type=StepType.PARALLEL_TASKS,
                agent_roles=["analyst", "coordinator"],
                instructions="Plan required resources",
                parameters={"tasks": "{{task_breakdown.results}}"},
                parallel_tasks=[
                    {"agent_role": "analyst", "task": "Resource requirements analysis"},
                    {"agent_role": "coordinator", "task": "Team and skill requirements"}
                ]
            ),
            WorkflowStep(
                step_id="timeline_creation",
                name="Timeline Creation",
                type=StepType.AGENT_TASK,
                agent_roles=["coordinator"],
                instructions="Create implementation timeline",
                parameters={
                    "tasks": "{{task_breakdown.results}}",
                    "resources": "{{resource_planning.results}}"
                }
            ),
            WorkflowStep(
                step_id="risk_mitigation",
                name="Risk Mitigation",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Identify and plan for implementation risks",
                parameters={"implementation_plan": "{{timeline_creation.results}}"}
            ),
            WorkflowStep(
                step_id="success_metrics",
                name="Success Metrics",
                type=StepType.AGENT_TASK,
                agent_roles=["analyst"],
                instructions="Define success metrics and monitoring",
                parameters={"plan": "{{timeline_creation.results}}"}
            )
        ]

        workflow_def = WorkflowDefinition(
            workflow_id="implementation_planning_template",
            name="Implementation Planning",
            description="Comprehensive implementation planning",
            steps=steps
        )

        template = WorkflowTemplate(
            name="Implementation Planning",
            description="Systematic planning for solution implementation with resources, timeline, and risk management",
            category=TemplateCategory.COORDINATION,
            difficulty=TemplateDifficulty.ADVANCED,
            parameters=[
                TemplateParameter(
                    name="solution_design",
                    description="The solution design to implement",
                    type="string",
                    required=True
                )
            ],
            workflow_definition=workflow_def,
            tags=["implementation", "planning", "coordination"],
            use_cases=["Project management", "Solution deployment", "Change management"],
            estimated_duration_minutes=90
        )

        self.library.add_template(template)

    async def create_workflow_from_template(self,
                                          template_id: str,
                                          parameters: Dict[str, Any],
                                          customizations: Optional[Dict[str, Any]] = None) -> WorkflowDefinition:
        """Create a workflow instance from a template"""
        if template_id not in self.library.templates:
            raise ValueError(f"Template {template_id} not found")

        template = self.library.templates[template_id]

        # Validate required parameters
        required_params = [p.name for p in template.parameters if p.required]
        missing_params = set(required_params) - set(parameters.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Create workflow with parameter substitution
        workflow_def = template.workflow_definition.model_copy(deep=True)
        workflow_def.workflow_id = str(uuid4())
        workflow_def.name = f"{template.name} - {datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Apply parameter substitution
        self._apply_parameter_substitution(workflow_def, parameters)

        # Apply customizations if provided
        if customizations:
            self._apply_customizations(workflow_def, customizations, template.customization)

        # Update usage count
        template.usage_count += 1
        template.updated_at = datetime.utcnow()

        logger.info(f"Created workflow from template {template.name}")
        return workflow_def

    def _apply_parameter_substitution(self, workflow_def: WorkflowDefinition, parameters: Dict[str, Any]) -> None:
        """Apply parameter substitution to workflow definition"""
        def substitute_value(value: Any) -> Any:
            if isinstance(value, str):
                for param_name, param_value in parameters.items():
                    placeholder = f"{{{{{param_name}}}}}"
                    if placeholder in value:
                        value = value.replace(placeholder, str(param_value))
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            return value

        # Apply substitution to all steps
        for step in workflow_def.steps:
            step.parameters = substitute_value(step.parameters)
            step.instructions = substitute_value(step.instructions)

            if step.parallel_tasks:
                step.parallel_tasks = substitute_value(step.parallel_tasks)
            if step.sequential_tasks:
                step.sequential_tasks = substitute_value(step.sequential_tasks)

    def _apply_customizations(self, workflow_def: WorkflowDefinition, customizations: Dict[str, Any], template_customization: TemplateCustomization) -> None:
        """Apply customizations to workflow definition"""
        if not template_customization.allow_step_modification and 'steps' in customizations:
            raise ValueError("Step modification not allowed for this template")

        if not template_customization.allow_condition_modification and 'conditions' in customizations:
            raise ValueError("Condition modification not allowed for this template")

        # Apply allowed customizations
        if template_customization.allow_step_modification and 'steps' in customizations:
            # Allow step modifications within limits
            pass

        if template_customization.allow_parameter_override and 'parameters' in customizations:
            # Allow parameter overrides
            pass

    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a template by ID"""
        return self.library.templates.get(template_id)

    def search_templates(self, **kwargs) -> List[WorkflowTemplate]:
        """Search templates"""
        return self.library.search_templates(**kwargs)

    def get_templates_by_category(self, category: TemplateCategory) -> List[WorkflowTemplate]:
        """Get templates by category"""
        return self.library.get_templates_by_category(category)

    def save_library(self, file_path: str) -> None:
        """Save template library to file"""
        try:
            library_data = self.library.model_dump()
            with open(file_path, 'w') as f:
                json.dump(library_data, f, indent=2, default=str)
            logger.info(f"Saved template library to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save template library: {e}")
            raise

    def load_library(self, file_path: str) -> None:
        """Load template library from file"""
        try:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    library_data = json.load(f)
                self.library = TemplateLibrary(**library_data)
                logger.info(f"Loaded template library from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load template library: {e}")
            raise

    def add_custom_template(self, template: WorkflowTemplate) -> None:
        """Add a custom template to the library"""
        self.library.add_template(template)

    def remove_template(self, template_id: str) -> bool:
        """Remove a template from the library"""
        return self.library.remove_template(template_id)

    def get_template_statistics(self) -> Dict[str, Any]:
        """Get statistics about the template library"""
        templates = list(self.library.templates.values())

        stats = {
            "total_templates": len(templates),
            "by_category": {},
            "by_difficulty": {},
            "most_used": [],
            "average_duration": 0,
            "total_usage": sum(t.usage_count for t in templates)
        }

        # Category breakdown
        for category in TemplateCategory:
            count = len([t for t in templates if t.category == category])
            if count > 0:
                stats["by_category"][category.value] = count

        # Difficulty breakdown
        for difficulty in TemplateDifficulty:
            count = len([t for t in templates if t.difficulty == difficulty])
            if count > 0:
                stats["by_difficulty"][difficulty.value] = count

        # Most used templates
        sorted_templates = sorted(templates, key=lambda x: x.usage_count, reverse=True)
        stats["most_used"] = [
            {"name": t.name, "usage_count": t.usage_count}
            for t in sorted_templates[:5]
        ]

        # Average duration
        durations = [t.estimated_duration_minutes for t in templates if t.estimated_duration_minutes]
        if durations:
            stats["average_duration"] = sum(durations) / len(durations)

        return stats