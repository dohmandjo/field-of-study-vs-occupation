"""Single source of truth for the dataset columns.

Every feature has three names:
  * ``csv``   - the column header in ``data/raw/field_of_study_vs_occupation.csv``
  * ``model`` - the snake_case column used inside the Spark pipeline
  * ``api``   - the camelCase JSON field used by the FastAPI service and the React form

Keeping the mapping in one place is what prevents the training code and the API from drifting apart.
"""
from __future__ import annotations

from dataclasses import dataclass

LABEL_CSV = "Likely to Change Occupation"
LABEL = "likely_to_change_occupation"


@dataclass(frozen=True)
class Feature:
    csv: str
    model: str
    api: str
    kind: str  # "categorical" | "numeric"


def _f(csv: str, api: str, kind: str) -> Feature:
    model = csv.lower().replace("-", "_").replace(" ", "_")
    return Feature(csv, model, api, kind)


FEATURES: tuple[Feature, ...] = (
    _f("Field of Study", "fieldOfStudy", "categorical"),
    _f("Current Occupation", "currentOccupation", "categorical"),
    _f("Age", "age", "numeric"),
    _f("Gender", "gender", "categorical"),
    _f("Years of Experience", "yearsOfExperience", "numeric"),
    _f("Education Level", "educationLevel", "categorical"),
    _f("Industry Growth Rate", "industryGrowthRate", "categorical"),
    _f("Job Satisfaction", "jobSatisfaction", "numeric"),
    _f("Work-Life Balance", "workLifeBalance", "numeric"),
    _f("Job Opportunities", "jobOpportunities", "numeric"),
    _f("Salary", "salary", "numeric"),
    _f("Job Security", "jobSecurity", "numeric"),
    _f("Career Change Interest", "careerChangeInterest", "numeric"),
    _f("Skills Gap", "skillsGap", "numeric"),
    _f("Family Influence", "familyInfluence", "categorical"),
    _f("Mentorship Available", "mentorshipAvailable", "numeric"),
    _f("Certifications", "certifications", "numeric"),
    _f("Freelancing Experience", "freelancingExperience", "numeric"),
    _f("Geographic Mobility", "geographicMobility", "numeric"),
    _f("Professional Networks", "professionalNetworks", "numeric"),
    _f("Career Change Events", "careerChangeEvents", "numeric"),
    _f("Technology Adoption", "technologyAdoption", "numeric"),
)

CATEGORICAL = tuple(f for f in FEATURES if f.kind == "categorical")
NUMERIC = tuple(f for f in FEATURES if f.kind == "numeric")

# Career Change Interest is (almost) the label in disguise: in the data every row with interest = 1 has
# label = 1. Training can drop it with ``--exclude careerChangeInterest``.
LEAKY_FEATURE = "careerChangeInterest"

BY_API = {f.api: f for f in FEATURES}
BY_MODEL = {f.model: f for f in FEATURES}
