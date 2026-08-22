-- Research Radar organisation watchlist v2 (30 orgs) + topic seed.
-- Replaces prior organisation seed; keeps topics.
BEGIN;

-- Deactivate previous seed orgs not in the new watchlist (safe for FK history).
UPDATE research_radar.organisations
SET active = FALSE, modified_at = NOW()
WHERE canonical_name IN (
  'Google',
  'University of Oxford',
  'University of Cambridge',
  'ETH Zurich',
  'Allen Institute for AI',
  'MIT',
  'UC Berkeley',
  'Carnegie Mellon University',
  'xAI',
  'Mistral AI',
  'Stanford University'
);

INSERT INTO research_radar.organisations(
  org_key, canonical_name, organisation_type, aliases, domains, priority, active,
  watchlist_tags, evidence_sources, rationale, recent_highlight, notes
) VALUES
(
  'openai', 'OpenAI', 'frontier_ai_lab',
  ARRAY['OpenAI Research'],
  ARRAY['openai.com'],
  10, TRUE,
  ARRAY['LLMs','agents','reasoning','coding','safety','evaluations'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','company blog'],
  'Must-watch frontier lab spanning foundation models, reasoning, coding agents, evaluations and safety.',
  'EVMbench — benchmark for AI agents finding/patching/exploiting smart-contract vulnerabilities (Feb 2026)',
  'watchlist_v2'
),
(
  'anthropic', 'Anthropic', 'frontier_ai_lab',
  ARRAY['Anthropic PBC'],
  ARRAY['anthropic.com'],
  10, TRUE,
  ARRAY['LLMs','agents','safety','alignment','interpretability','coding'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','company blog'],
  'Highest-signal source for frontier-model, alignment, interpretability, agent and safety research.',
  'Trustworthy agents in practice — agent governance, autonomy and prompt-injection risk (Apr 2026)',
  'watchlist_v2'
),
(
  'google-deepmind', 'Google DeepMind', 'frontier_ai_lab',
  ARRAY['DeepMind'],
  ARRAY['deepmind.google','deepmind.com'],
  10, TRUE,
  ARRAY['LLMs','agents','multimodal','reasoning','safety','robotics'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','company blog'],
  'Frontier multimodal/reasoning models plus agents, robotics, science and safety research.',
  'Gram — automated alignment auditing for sabotage in AI agents (May 2026)',
  'watchlist_v2'
),
(
  'deepseek', 'DeepSeek', 'frontier_ai_lab',
  ARRAY['DeepSeek AI','DeepSeek-AI'],
  ARRAY['deepseek.com'],
  10, TRUE,
  ARRAY['LLMs','reasoning','open-weights','training','efficiency','coding'],
  ARRAY['official technical reports','arXiv','OpenAlex','Google Scholar','official model page'],
  'New model generations/technical reports can shift assumptions on training efficiency and open-model competitiveness.',
  'DeepSeek-V4 — model card and technical report listed in Transparency Center (Apr 2026)',
  'watchlist_v2'
),
(
  'mistral-ai', 'Mistral AI', 'frontier_ai_lab',
  ARRAY['Mistral'],
  ARRAY['mistral.ai'],
  10, TRUE,
  ARRAY['LLMs','agents','open-weights','safety','multimodal','physics-AI'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','company blog'],
  'European frontier lab spanning open-weight models, agents, retrieval, coding, multimodality and safety.',
  'Shieldstral — open-weight multimodal policy-adaptive safety classifier (Aug 2026)',
  'watchlist_v2'
),
(
  'xai', 'xAI', 'frontier_ai_lab',
  ARRAY['SpaceXAI'],
  ARRAY['x.ai'],
  8, TRUE,
  ARRAY['LLMs','agents','coding','reasoning','multimodal','tool-use'],
  ARRAY['official model page','arXiv','OpenAlex','Google Scholar','company blog'],
  'Frontier-model watch for reasoning, coding, knowledge work, multimodal interfaces and agents.',
  'Grok 4.5 — model focused on coding, agentic tasks and knowledge work (Jul 2026)',
  'watchlist_v2'
),
(
  'meta', 'Meta', 'big_tech',
  ARRAY['Meta AI','FAIR'],
  ARRAY['meta.com','ai.meta.com','fb.com'],
  10, TRUE,
  ARRAY['LLMs','agents','RAG','multimodal','open-source','RL'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','company blog'],
  'Major research source across core ML, agents, retrieval, multimodality, RL, vision and open systems.',
  'AIRA₂ — architecture for stronger AI research agents (Apr 2026)',
  'watchlist_v2'
),
(
  'microsoft', 'Microsoft', 'big_tech',
  ARRAY['Microsoft Research','MSR','Microsoft AI'],
  ARRAY['microsoft.com'],
  10, TRUE,
  ARRAY['agents','LLMs','RAG','memory','enterprise-AI','evaluations'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','company blog'],
  'Microsoft Research publishes across models, agents, memory, retrieval, systems and enterprise workflows.',
  'CORPGEN — autonomous digital employees in multi-horizon corporate task environments (Feb 2026)',
  'watchlist_v2'
),
(
  'amazon', 'Amazon', 'big_tech',
  ARRAY['Amazon Science','Amazon AGI','AWS','Amazon Web Services'],
  ARRAY['amazon.com','amazon.science','aws.amazon.com'],
  10, TRUE,
  ARRAY['agents','LLMs','multimodal','robotics','RAG','enterprise-AI'],
  ARRAY['Amazon Science','arXiv','OpenAlex','Google Scholar','AWS/company blog'],
  'Broad scientific publishing with AGI, AWS, robotics and production-scale applied ML affiliations.',
  'Perception agent harness — open-source annotation/verification primitives for multimodal agents (May 2026)',
  'watchlist_v2'
),
(
  'apple', 'Apple', 'big_tech',
  ARRAY['Apple Machine Learning Research','Apple ML Research'],
  ARRAY['apple.com','machinelearning.apple.com'],
  8, TRUE,
  ARRAY['LLMs','multimodal','on-device','privacy','agents','efficient-inference'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','company research blog'],
  'On-device models, multimodality, privacy-preserving inference, efficient architectures and agentic tool use.',
  'Third Generation Apple Foundation Models — on-device and cloud models including agentic tool use (Jun 2026)',
  'watchlist_v2'
),
(
  'nvidia', 'NVIDIA', 'ai_infrastructure',
  ARRAY['NVIDIA Research','Nvidia'],
  ARRAY['nvidia.com','research.nvidia.com'],
  10, TRUE,
  ARRAY['LLMs','training','RL','inference','agents','multimodal'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','company blog'],
  'Foundational work in model training, reasoning, RL, multimodality, robotics and inference.',
  'RLP: Reinforcement as a Pretraining Objective — ICLR 2026',
  'watchlist_v2'
),
(
  'hugging-face', 'Hugging Face', 'ai_infrastructure',
  ARRAY['HF','HuggingFace'],
  ARRAY['huggingface.co'],
  8, TRUE,
  ARRAY['open-source','agents','RL','RAG','models','datasets'],
  ARRAY['official blog','Hugging Face Hub','arXiv','OpenAlex','Google Scholar'],
  'Ecosystem signal for models, datasets, evaluation and open agent infrastructure.',
  'OpenEnv — community-backed interoperability layer for agentic RL environments (Jun 2026)',
  'watchlist_v2'
),
(
  'databricks', 'Databricks', 'ai_infrastructure',
  ARRAY['Databricks AI'],
  ARRAY['databricks.com'],
  8, TRUE,
  ARRAY['agents','RAG','data-science','MLOps','evaluation','enterprise-AI'],
  ARRAY['official documentation','official research/blog','arXiv','OpenAlex','Google Scholar'],
  'Applied signal at intersection of enterprise data, RAG, evaluation, governance and agents.',
  'Agent Bricks 2026 expansion — agents, document intelligence and long-running supervisor workflows',
  'watchlist_v2'
),
(
  'cerebras', 'Cerebras', 'ai_infrastructure',
  ARRAY['Cerebras Systems'],
  ARRAY['cerebras.ai'],
  3, TRUE,
  ARRAY['inference','AI-infrastructure','agents','LLMs','hardware'],
  ARRAY['official engineering blog','official research page','arXiv','OpenAlex','Google Scholar'],
  'Specialist watch for inference speed, agent latency and novel compute architecture stories.',
  'OpenAI–Cerebras high-speed inference partnership — 750 MW deployment (Jan 2026)',
  'watchlist_v2'
),
(
  'ai2', 'Ai2', 'research_lab',
  ARRAY['Allen Institute for AI','Allen AI','AI2'],
  ARRAY['allenai.org'],
  8, TRUE,
  ARRAY['agents','scientific-AI','evaluations','open-source','LLMs','robotics'],
  ARRAY['official research page','arXiv','OpenAlex','Semantic Scholar','Google Scholar'],
  'Independent source for open AI research, scientific agents, evaluation, NLP and scholarly infrastructure.',
  'AstaBench 2026 update — open benchmark for scientific-research agents',
  'watchlist_v2'
),
(
  'ibm-research', 'IBM Research', 'research_lab',
  ARRAY['IBM'],
  ARRAY['research.ibm.com','ibm.com'],
  6, TRUE,
  ARRAY['agents','evaluations','enterprise-AI','RAG','governance','LLMs'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','company blog'],
  'Enterprise-grade AI research on agents, governance, systems and production evaluation.',
  'Measuring Agents in Production — systematic study of production agent development (ICLR 2026)',
  'watchlist_v2'
),
(
  'mila', 'Mila', 'research_lab',
  ARRAY['Quebec AI Institute','Mila - Quebec AI Institute'],
  ARRAY['mila.quebec'],
  6, TRUE,
  ARRAY['safety','alignment','LLMs','agents','NLP','RL'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','lab/faculty pages'],
  'High-density source of fundamental ML, NLP, safety and alignment research.',
  'Mila research focus includes safer and more transparent agentic LLMs (2026)',
  'watchlist_v2'
),
(
  'stanford-university', 'Stanford University', 'university',
  ARRAY['Stanford','Stanford HAI','Stanford AI Lab','SAIL'],
  ARRAY['stanford.edu','hai.stanford.edu','cs.stanford.edu'],
  8, TRUE,
  ARRAY['agents','LLMs','RAG','human-AI','evaluations','ML-systems'],
  ARRAY['official lab pages','arXiv','OpenAlex','Google Scholar','Stanford research pages'],
  'High-yield source for foundation-model, agent, human-AI, evaluation and ML-systems research.',
  'AgentFlow — in-the-flow agentic system optimization for planning and tool use (ICLR 2026)',
  'watchlist_v2'
),
(
  'mit', 'Massachusetts Institute of Technology', 'university',
  ARRAY['MIT','MIT CSAIL','CSAIL'],
  ARRAY['mit.edu','csail.mit.edu'],
  8, TRUE,
  ARRAY['agents','robotics','ML-systems','LLMs','multimodal','safety'],
  ARRAY['official lab pages','arXiv','OpenAlex','Google Scholar','MIT News'],
  'Research spanning agents, robotics, systems, multimodal AI, safety and AI for science.',
  'SceneSmith — collaborative AI agents generate training environments for robots (Jul 2026)',
  'watchlist_v2'
),
(
  'uc-berkeley', 'University of California, Berkeley', 'university',
  ARRAY['UC Berkeley','Berkeley','BAIR','Berkeley AI Research'],
  ARRAY['berkeley.edu','bair.berkeley.edu'],
  8, TRUE,
  ARRAY['agents','data-systems','LLMs','RAG','RL','open-source'],
  ARRAY['BAIR/official lab pages','arXiv','OpenAlex','Google Scholar','university research pages'],
  'Foundational ML with open-source tooling, data systems, agents and reinforcement learning.',
  'Data Systems for, of, and by Agents — architecture agenda for agent-native data systems (Jul 2026)',
  'watchlist_v2'
),
(
  'carnegie-mellon', 'Carnegie Mellon University', 'university',
  ARRAY['CMU','Carnegie Mellon'],
  ARRAY['cmu.edu','cs.cmu.edu'],
  6, TRUE,
  ARRAY['agents','robotics','ML-systems','multimodal','NLP','AI-for-science'],
  ARRAY['official lab pages','arXiv','OpenAlex','Google Scholar','university research pages'],
  'Deep coverage across robotics, language technologies, agents, multimodal systems and AI infrastructure.',
  'Autonomous laboratories — AI agents plus digital twins for scientific automation (Jul 2026)',
  'watchlist_v2'
),
(
  'tsinghua-university', 'Tsinghua University', 'university',
  ARRAY['Tsinghua','AIR Tsinghua','Institute for AI Industry Research'],
  ARRAY['tsinghua.edu.cn','air.tsinghua.edu.cn'],
  6, TRUE,
  ARRAY['multimodal','agents','AI-for-science','healthcare','LLMs','ML'],
  ARRAY['official university/lab pages','arXiv','OpenAlex','Google Scholar'],
  'Essential non-US-centric source across core ML, multimodal AI, agents and AI for science.',
  'UniAIR — unified multimodal mutation-effect prediction (Nature Machine Intelligence 2026)',
  'watchlist_v2'
),
(
  'cohere', 'Cohere', 'startup',
  ARRAY['Cohere Labs','Cohere For AI'],
  ARRAY['cohere.com'],
  8, TRUE,
  ARRAY['LLMs','agents','RAG','multilingual','enterprise-AI','RL'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','company blog'],
  'Enterprise agent/RAG deployment plus open research on RL, evaluation and multilingual models.',
  'CIRCLE, Soft-SVeRL and Tiny Aya — representative 2026 Cohere Labs research',
  'watchlist_v2'
),
(
  'sakana-ai', 'Sakana AI', 'startup',
  ARRAY['Sakana'],
  ARRAY['sakana.ai'],
  8, TRUE,
  ARRAY['agents','multi-agent','orchestration','RL','AI-scientist','LLMs'],
  ARRAY['official research page','arXiv','OpenAlex','Google Scholar','company blog'],
  'High-signal emerging lab for autonomous research, collective intelligence and multi-agent orchestration.',
  'Learning to Orchestrate Agents in Natural Language with the Conductor (ICLR 2026)',
  'watchlist_v2'
),
(
  'thinking-machines-lab', 'Thinking Machines Lab', 'startup',
  ARRAY['Thinking Machines'],
  ARRAY['thinkingmachines.ai'],
  8, TRUE,
  ARRAY['LLMs','multimodal','agents','human-AI','RL','open-weights'],
  ARRAY['official research/blog','model cards','arXiv','OpenAlex','Google Scholar'],
  'Emerging source for frontier models, multimodal interaction, model customization and agentic capabilities.',
  'Interaction Models — scalable real-time multimodal human-AI collaboration (May 2026)',
  'watchlist_v2'
),
(
  'nist-caisi', 'NIST Center for AI Standards and Innovation', 'government',
  ARRAY['NIST','CAISI','National Institute of Standards and Technology','Center for AI Standards and Innovation'],
  ARRAY['nist.gov'],
  8, TRUE,
  ARRAY['safety','standards','agents','evaluations','security','governance'],
  ARRAY['official research page','NIST publications','official standards/guidelines','arXiv','Google Scholar'],
  'Technical evaluation, AI security, measurement science and standards for agents and AI systems.',
  'AI Agent Standards Initiative — interoperability and security initiative (Feb 2026)',
  'watchlist_v2'
),
(
  'uk-ai-security-institute', 'UK AI Security Institute', 'government',
  ARRAY['UK AISI','AI Security Institute UK'],
  ARRAY['aisi.gov.uk'],
  8, TRUE,
  ARRAY['safety','agents','evaluations','security','MCP','policy'],
  ARRAY['official research page','official publications','arXiv','OpenAlex','Google Scholar'],
  'Empirical research on frontier-system evaluation, agent behavior and real-world AI security.',
  'How are AI agents used? Evidence from 177,000 MCP tools (2026)',
  'watchlist_v2'
),
(
  'mlcommons', 'MLCommons', 'standards_body',
  ARRAY['MLCommons Association','MLPerf'],
  ARRAY['mlcommons.org'],
  8, TRUE,
  ARRAY['evaluations','benchmarks','safety','security','inference','LLMs'],
  ARRAY['official benchmark pages','official technical reports','arXiv','OpenAlex','Google Scholar'],
  'Independent benchmarking/measurement consortium providing cross-vendor evidence.',
  'AILuminate — shared safety/security benchmark family for generative AI',
  'watchlist_v2'
),
(
  'iso-iec-jtc1-sc42', 'ISO/IEC JTC 1/SC 42', 'standards_body',
  ARRAY['SC 42','JTC 1 SC 42','ISO AI Committee'],
  ARRAY['iso.org'],
  3, TRUE,
  ARRAY['AI-standards','evaluations','testing','governance','ML-efficiency','human-AI'],
  ARRAY['official standards catalogue','official committee page','technical reports','OpenAlex','Google Scholar'],
  'Specialist monitoring for AI practices becoming durable international standards.',
  'ISO/IEC TR 42106:2026 — differentiated benchmarking of AI-system quality (Jul 2026)',
  'watchlist_v2'
),
(
  'ieee-standards-association', 'IEEE Standards Association', 'standards_body',
  ARRAY['IEEE SA','IEEE'],
  ARRAY['standards.ieee.org','ieee.org'],
  3, TRUE,
  ARRAY['AI-standards','agents','evaluations','interoperability','AI-literacy','multimodal'],
  ARRAY['official standards catalogue','IEEE publications','OpenAlex','Google Scholar','arXiv'],
  'Specialist monitoring for AI-agent evaluation, interoperability and multimodal interaction standards.',
  'IEEE P3777 — Standard for Benchmarking and Performance Metrics of AI Agents',
  'watchlist_v2'
)
ON CONFLICT (canonical_name) DO UPDATE SET
  org_key = EXCLUDED.org_key,
  organisation_type = EXCLUDED.organisation_type,
  aliases = EXCLUDED.aliases,
  domains = EXCLUDED.domains,
  priority = EXCLUDED.priority,
  active = TRUE,
  watchlist_tags = EXCLUDED.watchlist_tags,
  evidence_sources = EXCLUDED.evidence_sources,
  rationale = EXCLUDED.rationale,
  recent_highlight = EXCLUDED.recent_highlight,
  notes = EXCLUDED.notes,
  modified_at = NOW();

-- Ensure Ai2 alias covers previous Allen Institute row if still present under old name.
UPDATE research_radar.organisations
SET active = FALSE, modified_at = NOW()
WHERE canonical_name = 'Allen Institute for AI';

INSERT INTO research_radar.topics(canonical_name,aliases) VALUES
('AI agents',ARRAY['agentic systems','autonomous agents','multi-agent','tool use']),
('LLMs / foundation models',ARRAY['large language model','foundation model','LLM']),
('Machine learning',ARRAY['ML']),
('Data science',ARRAY['data analytics']),
('Multimodal AI',ARRAY['vision-language','VLM']),
('AI engineering / ML systems',ARRAY['MLOps','LLMOps','inference systems']),
('AI evaluation',ARRAY['evals','benchmarking','evaluation']),
('AI safety and security',ARRAY['AI security','alignment','model safety']),
('Retrieval / RAG',ARRAY['retrieval augmented generation','RAG']),
('Model training, inference and efficiency',ARRAY['training efficiency','inference optimization','quantization']),
('Applied AI',ARRAY['AI application']),
('Human-AI interaction',ARRAY['HCI','human AI']),
('AI product and experimentation',ARRAY['AI product','experimentation','A/B testing'])
ON CONFLICT (canonical_name) DO UPDATE SET aliases=EXCLUDED.aliases;

COMMIT;
