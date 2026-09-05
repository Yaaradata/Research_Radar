-- Closed vocabulary for the topics stage: domains, subdomains (linked to
-- their domain) and applications. domain/subdomain/application are CLOSED —
-- the topics prompt supplies exactly these lists and anything the model
-- returns outside them is dropped in post-processing (topics.py).
--
-- 'Other' is included as a 13th domain because the topics-v1 system prompt
-- explicitly allows the model to return "Other" when no domain fits; without
-- a seeded row for it, that fallback would fail closed-vocabulary validation
-- and every off-taxonomy paper would silently lose its domain.

BEGIN;

INSERT INTO research_radar.topics (canonical_name, level, origin, aliases) VALUES
    ('Natural Language Processing',   'domain', 'seed', ARRAY['NLP']),
    ('Computer Vision',               'domain', 'seed', ARRAY['CV']),
    ('Reinforcement Learning',        'domain', 'seed', ARRAY['RL']),
    ('Machine Learning Theory',       'domain', 'seed', ARRAY[]::TEXT[]),
    ('Speech and Audio',              'domain', 'seed', ARRAY[]::TEXT[]),
    ('Robotics',                      'domain', 'seed', ARRAY[]::TEXT[]),
    ('Information Retrieval',         'domain', 'seed', ARRAY['IR']),
    ('AI Safety and Alignment',       'domain', 'seed', ARRAY[]::TEXT[]),
    ('ML Systems and Efficiency',     'domain', 'seed', ARRAY[]::TEXT[]),
    ('Multimodal Learning',           'domain', 'seed', ARRAY[]::TEXT[]),
    ('Human-AI Interaction',          'domain', 'seed', ARRAY['HCI']),
    ('Generative Modelling',          'domain', 'seed', ARRAY['Generative Modeling']),
    ('Other',                         'domain', 'seed', ARRAY[]::TEXT[])
ON CONFLICT (canonical_name) DO UPDATE SET
    level = EXCLUDED.level,
    origin = EXCLUDED.origin,
    aliases = EXCLUDED.aliases;

INSERT INTO research_radar.topics (canonical_name, level, origin, parent_topic_id)
SELECT v.name, 'subdomain', 'seed', d.topic_id
FROM (VALUES
    -- Natural Language Processing
    ('Text Classification',                 'Natural Language Processing'),
    ('AI-Generated Content Detection',      'Natural Language Processing'),
    ('Machine Translation',                 'Natural Language Processing'),
    ('Question Answering',                  'Natural Language Processing'),
    ('Named Entity Recognition',            'Natural Language Processing'),
    ('Sentiment Analysis',                  'Natural Language Processing'),
    ('Text Summarization',                  'Natural Language Processing'),
    ('Dialogue Systems',                    'Natural Language Processing'),
    ('Natural Language Inference',          'Natural Language Processing'),
    -- Computer Vision
    ('Object Detection',                    'Computer Vision'),
    ('Image Classification',                'Computer Vision'),
    ('Image Segmentation',                  'Computer Vision'),
    ('Visual Question Answering',           'Computer Vision'),
    ('Pose Estimation',                     'Computer Vision'),
    ('Video Understanding',                 'Computer Vision'),
    ('3D Vision and Reconstruction',        'Computer Vision'),
    -- Reinforcement Learning
    ('Deep Reinforcement Learning',         'Reinforcement Learning'),
    ('Multi-Agent Reinforcement Learning',  'Reinforcement Learning'),
    ('Offline Reinforcement Learning',      'Reinforcement Learning'),
    ('Reward Modelling',                    'Reinforcement Learning'),
    ('Exploration Strategies',              'Reinforcement Learning'),
    -- Machine Learning Theory
    ('Optimization Theory',                 'Machine Learning Theory'),
    ('Generalization and Learning Theory',  'Machine Learning Theory'),
    ('Statistical Learning',                'Machine Learning Theory'),
    ('Causal Inference',                    'Machine Learning Theory'),
    ('Representation Learning',             'Machine Learning Theory'),
    ('Federated Learning',                  'Machine Learning Theory'),
    ('Benchmarking and Evaluation',         'Machine Learning Theory'),
    -- Speech and Audio
    ('Automatic Speech Recognition',        'Speech and Audio'),
    ('Text-to-Speech Synthesis',            'Speech and Audio'),
    ('Speaker Identification',              'Speech and Audio'),
    ('Music Generation and Analysis',       'Speech and Audio'),
    -- Robotics
    ('Robot Manipulation',                  'Robotics'),
    ('Robot Navigation and SLAM',           'Robotics'),
    ('Sim-to-Real Transfer',                'Robotics'),
    ('Human-Robot Interaction',             'Robotics'),
    -- Information Retrieval
    ('RAG',                                 'Information Retrieval'),
    ('Dense Retrieval',                     'Information Retrieval'),
    ('Recommender Systems',                 'Information Retrieval'),
    ('Search Ranking',                      'Information Retrieval'),
    ('Knowledge Graphs',                    'Information Retrieval'),
    -- AI Safety and Alignment
    ('Interpretability',                    'AI Safety and Alignment'),
    ('Adversarial Robustness',              'AI Safety and Alignment'),
    ('Alignment and RLHF',                  'AI Safety and Alignment'),
    ('Fairness and Bias',                   'AI Safety and Alignment'),
    ('Privacy-Preserving Machine Learning', 'AI Safety and Alignment'),
    ('AI Governance and Policy',            'AI Safety and Alignment'),
    -- ML Systems and Efficiency
    ('Model Compression',                   'ML Systems and Efficiency'),
    ('Efficient Inference',                 'ML Systems and Efficiency'),
    ('Distributed Training',                'ML Systems and Efficiency'),
    ('Hardware-Aware ML',                   'ML Systems and Efficiency'),
    ('Model Serving and Deployment',        'ML Systems and Efficiency'),
    -- Multimodal Learning
    ('Vision-Language Models',              'Multimodal Learning'),
    ('Multimodal Foundation Models',        'Multimodal Learning'),
    ('Cross-Modal Retrieval',               'Multimodal Learning'),
    ('Embodied AI',                         'Multimodal Learning'),
    -- Human-AI Interaction
    ('Agents and Tool Use',                 'Human-AI Interaction'),
    ('Human-in-the-Loop Learning',          'Human-AI Interaction'),
    ('Explainable AI',                      'Human-AI Interaction'),
    ('Prompt Engineering',                  'Human-AI Interaction'),
    ('AI-Assisted Programming',             'Human-AI Interaction'),
    -- Generative Modelling
    ('Diffusion Models',                    'Generative Modelling'),
    ('Autoregressive Generation',           'Generative Modelling'),
    ('Generative Adversarial Networks',     'Generative Modelling'),
    ('Synthetic Data Generation',           'Generative Modelling')
) AS v(name, domain)
JOIN research_radar.topics d ON d.canonical_name = v.domain AND d.level = 'domain'
ON CONFLICT (canonical_name) DO UPDATE SET
    level = 'subdomain',
    origin = 'seed',
    parent_topic_id = EXCLUDED.parent_topic_id;

INSERT INTO research_radar.topics (canonical_name, level, origin) VALUES
    ('education',             'application', 'seed'),
    ('healthcare',            'application', 'seed'),
    ('finance',               'application', 'seed'),
    ('legal',                 'application', 'seed'),
    ('academic-integrity',    'application', 'seed'),
    ('software-engineering',  'application', 'seed'),
    ('security',              'application', 'seed'),
    ('scientific-discovery',  'application', 'seed'),
    ('accessibility',         'application', 'seed'),
    ('content-moderation',    'application', 'seed'),
    ('manufacturing',         'application', 'seed'),
    ('agriculture',           'application', 'seed'),
    ('customer-service',      'application', 'seed'),
    ('marketing',             'application', 'seed'),
    ('journalism',            'application', 'seed'),
    ('government-and-policy', 'application', 'seed'),
    ('transportation',        'application', 'seed'),
    ('energy',                'application', 'seed'),
    ('gaming',                'application', 'seed'),
    ('defense',               'application', 'seed')
ON CONFLICT (canonical_name) DO UPDATE SET
    level = EXCLUDED.level,
    origin = EXCLUDED.origin;

COMMIT;
