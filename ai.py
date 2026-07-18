import torch
import json
import requests
from transformers import AutoTokenizer, AutoModel
from config import auth_headers, url

# Initialize the AI model
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")

def embed_text(text):
    """
    Embed the given text using a pre-trained model.
    @param text - The input text to be embedded
    @return The embedded representation of the input text
    """
    if text is None:
        text = "..."
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def generate_cluster_name(cluster):
    """
    Generate a cluster name based on the given cluster data by sending a request to a specified ChatGPT4 URL.
    @param cluster - the cluster data
    @return None
    """
    request_payload = {
        "messages": [
            {
                "role": "system",
                "content": "Give an appropriate name for the given cluster."
            },
            {
                "role": "user",
                "content": "Here is the data of new academic papers and their most similar but older papers. Given this information, please give a name for the cluster. Only give and say the name, no other tokens is needed from you. Only the name you found."
            },
            {
                "role": "user",
                "content": json.dumps(cluster)
            }
        ]
    }

    req = requests.post(url, headers=auth_headers, json=request_payload, verify=False)
    res = req.json()
    return f'"{res["choices"][0]["message"]["content"].strip()}"'

def identify_trends_in_cluster(cluster):
    """
    Identify trends in a specific cluster by sending a request to an ChatGPT4 API endpoint with the necessary payload of prompt engineering.
    @param cluster - the cluster to analyze
    @return None
    """
    request_payload = {
        "messages": [
            {
                "role": "system",
                "content": "Analyze the dataset consisting of new academic papers and their similar older counterparts. Identify trends, shifts in research focus, and notable thematic or methodological changes."
            },
            {
                "role": "user",
                "content": """Here is the data of new academic papers and their most similar but older papers. Given this information, please analyze and extract significant trends, research focus shifts, and thematic discontinuities. 
                The response should be of this form: 
                - <b>"Category of trend 1"</b>: ... / <b>Importance degree: "score"</b> <br>
                - <b>"Category of trend 2"</b>: ... / <b>Importance degree: "score"</b> <br>
                Do not add anything more to your response. No aknowledgment of the request, nothing but the format of response I have given to you.
                The importance degree is how important a trend is, depending on multiple factors as the need for immediate actions for example.
                The score of importance degree should be either: Low, Medium, High.
                Small means not really sure ; it's a possible trend.
                High means very important novelty or trend that needs immediate attention.
                Sort the list of trends on the importance degree score from the highest to the lowest.
                Each should be a small paragraph explaining the trend.
                I also want you to categorize the trends into these categories:
                -	An opposition: a new paper saying something opposite to an older paper
                -	A shift of focus on a specific domain (ex: weather / virus influence on wheat production)
                -	Emerging topic, a new topic, new field of study never seen before (ex: microbiome's impact on food safety)
                -	An increase of papers talking about the same topic
                -	Technological advances (ex: genomic techniques for pathogen detection or blockchain for traceability in the food supply chain)
                -	Regulatory Changes (ex : changes in food safety regulations)
                -	Geographical shifts: (ex: shift in focus towards certain regions, perhaps due to emerging food safety issues in those areas or new markets opening up)
                -	Methodological innovations: (ex: big data analytics, machine learning, AI…)
                -	Public Health Concerns: (ex: rising awareness of allergens or antimicrobial resistance)
                -	Risk Assessment and Management
                -	Consumer behavior and consumption
                -	Environmental impact
                -	Economic: funding trends for food safety research
                -	Cultural and Sociopolitical Factors: how different cultures approach food safety and how political stability affects food safety infrastructure.
"""
            },
            {
                "role": "user",
                "content": json.dumps(cluster)
            }
        ]
    }

    req = requests.post(url, headers=auth_headers, json=request_payload, verify=False)
    res = req.json()
    return res["choices"][0]["message"]["content"]
