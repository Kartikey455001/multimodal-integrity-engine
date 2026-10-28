from integrity_engine.preprocessing.chunking import chunk_segments
from integrity_engine.schemas import Segment

def test_empty_and_short_input():
    assert chunk_segments("d","f","pdf",[]) == []
    result=chunk_segments("d","f","pdf",[Segment("One sentence.",{"page":2})])
    assert len(result)==1 and result[0].location["page"]==2

def test_overlap_and_deterministic_ids():
    text=" ".join(f"Sentence {i} has enough words to form a sentence." for i in range(12))
    chunks=chunk_segments("d","f","docx",[Segment(text,{"paragraph":3})],25,10)
    assert len(chunks)>1 and chunks[0].chunk_id=="d::chunk-0000"
    assert all(c.location["paragraph"]==3 for c in chunks)
