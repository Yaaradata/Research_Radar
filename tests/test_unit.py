from research_radar.pipeline import extract_arxiv_id, normalize_url, score_relevance

def test_arxiv_version():
    assert extract_arxiv_id('https://arxiv.org/abs/2608.02345v2') == ('2608.02345',2)
    assert normalize_url('https://arxiv.org/pdf/2608.02345v1') == 'https://arxiv.org/abs/2608.02345'

def test_relevance():
    score,topic,_,_=score_relevance('Evaluating Tool-Using AI Agents','Benchmark for production agent systems.',['cs.AI'],'arxiv')
    assert score >= 5
    assert topic in {'AI agents','AI evaluation'}
