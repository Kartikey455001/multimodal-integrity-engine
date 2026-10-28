from integrity_engine.detection.ai_content import ai_content_indicator

def test_indicator_deterministic_and_bounded():
    text="A repeated regular sentence appears. A repeated regular sentence appears. "*20
    a=ai_content_indicator(text,10)
    assert a==ai_content_indicator(text,10) and 0<=a["score"]<=100

def test_empty_and_short_are_insufficient():
    assert ai_content_indicator("")["confidence"]=="insufficient"
    assert ai_content_indicator("Few words only.")["confidence"]=="insufficient"
