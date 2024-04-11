from sentence_transformers import SentenceTransformer
from sentence_transformers import util

def score_cos_sim(art1,art2):
    scores = util.cos_sim(art1, art2)[0]
    return scores

def score_inference(resume, job_description):
    # load the model
    model = SentenceTransformer('msmarco-distilbert-base-tas-b-final')
    
    score = dict()
    
    resume_encode = model.encode(resume)
    job_description_encode = model.encode(job_description)
    
    return score_cos_sim(resume_encode, job_description_encode)