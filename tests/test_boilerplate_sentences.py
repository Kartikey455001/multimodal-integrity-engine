from integrity_engine.detection.boilerplate import BoilerplateDetector
from integrity_engine.schemas import Document,Segment

def test_common_prompt_inside_independent_answers_is_detected():
    prompt="Explain your selected case study, describe the central problem, and recommend a practical solution."
    docs=[Document(str(i),str(i),"docx",segments=[Segment(prompt+f" My distinct analysis number {i} is useful.")]) for i in range(5)]
    detector=BoilerplateDetector(5).fit(docs)
    assert detector.is_boilerplate(prompt)
