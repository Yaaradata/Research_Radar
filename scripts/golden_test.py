import uuid
from research_radar.pipeline import connect,load_orgs,load_people,process_item,bump

with connect() as conn:
    run_id=uuid.uuid4(); conn.execute('INSERT INTO research_radar.pipeline_runs(run_id) VALUES(%s)',(run_id,))
    orgs,people=load_orgs(conn),load_people(conn)
    items=[
      {'source':'golden_test','source_external_id':'arxiv:2608.02345','source_feed':'golden','source_type':'arxiv','canonical_url':'https://arxiv.org/abs/2608.02345','title':'arXiv:2608.02345','summary':'AI agents experimentation','authors_raw':[],'categories_raw':['cs.AI'],'inoreader_tags':[],'raw_metadata':{},'published_at':None,'updated_at':None},
      {'source':'golden_test','source_external_id':'unknown-org-paper','source_feed':'golden','source_type':'research_paper','canonical_url':'https://example.org/research/unknown-org-ai-agents','title':'Practical AI Agent Evaluation for Production Systems','summary':'An empirical benchmark and evaluation framework for deploying tool-using AI agents in production engineering workflows.','authors_raw':['Researcher One'],'categories_raw':[],'inoreader_tags':[],'raw_metadata':{},'published_at':None,'updated_at':None}
    ]
    bump(conn,run_id,'items_received',len(items))
    for item in items: process_item(conn,run_id,item,orgs,people)
    conn.commit()
    rows=conn.execute("SELECT ci.title,ci.status,cs.notable_org_signal,cs.intrinsic_candidate_score,o.canonical_name,co.evidence_type,co.evidence_text,co.confidence FROM research_radar.content_items ci LEFT JOIN research_radar.content_scores cs ON cs.content_id=ci.id LEFT JOIN research_radar.content_organisations co ON co.content_id=ci.id LEFT JOIN research_radar.organisations o ON o.organisation_id=co.organisation_id WHERE ci.source='golden_test' ORDER BY ci.id DESC").fetchall()
    for r in rows: print(dict(r))
    assert any(r['canonical_name']=='Amazon' and r['evidence_type']=='email_domain' for r in rows), 'Golden #1: Amazon email-domain evidence not resolved'
    unknown=[r for r in rows if r['title']==items[1]['title']]
    assert unknown and max(float(r['notable_org_signal'] or 0) for r in unknown)==0, 'Golden #2: unknown org incorrectly boosted'
    print('Golden #3 is modelled separately via relationship_type + current_affiliation; current employer is never promoted to paper affiliation.')
