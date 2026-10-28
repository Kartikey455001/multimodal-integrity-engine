from integrity_engine.detection.boilerplate import BoilerplateDetector
from integrity_engine.schemas import Document,Segment

def test_repeated_text_is_marked_without_deleting_it():
    phrase="Please answer the fictional assignment prompt in your own words."
    docs=[Document(str(i),str(i),"docx",segments=[Segment(phrase)]) for i in range(5)]
    detector=BoilerplateDetector(5).fit(docs)
    assert detector.is_boilerplate(phrase) and all(d.segments for d in docs)
