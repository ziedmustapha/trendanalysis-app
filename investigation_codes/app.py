from flask import Flask, request, jsonify
import requests
import xml.etree.ElementTree as ET
import logging
import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import matplotlib.colors as mcolors
import numpy as np

# Import Nestlé credentials / URL from shared config (loaded from .env)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import auth_headers, url

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the AI model
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")

# PubMed fetching functions
def fetch_pubmed_data(query, mindate, maxdate, max_records):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    retmax = min(1000, max_records)
    retstart = 0
    all_ids = []

    while retstart < max_records:
        params = {
            'db': 'pubmed',
            'term': query,
            'mindate': mindate,
            'maxdate': maxdate,
            'retmode': 'xml',
            'retmax': retmax,
            'retstart': retstart
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        ids = [id_tag.text for id_tag in root.findall('.//Id')]
        
        if not ids:
            break

        all_ids.extend(ids)
        retstart += retmax

        if len(ids) < retmax:
            break  # Exit if there are no more records to fetch

    return all_ids

def fetch_details(pubmed_ids):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    retmax = 200
    all_details = []

    for i in range(0, len(pubmed_ids), retmax):
        batch_ids = pubmed_ids[i:i+retmax]
        params = {
            'db': 'pubmed',
            'id': ','.join(batch_ids),
            'retmode': 'xml'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        batch_xml = response.text
        batch_xml = batch_xml.split('?>', 1)[-1].strip()
        batch_xml = batch_xml.replace('<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">', '')
        all_details.append(batch_xml)

    combined_xml = "<root>" + "".join(all_details) + "</root>"
    return combined_xml

def parse_article_metadata(xml_data):
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        logging.error(f"Error parsing XML: {e}")
        return []

    articles = []
    for article in root.findall('.//PubmedArticle'):
        article_data = {}
        title_element = article.find('.//ArticleTitle')
        abstract_element = article.find('.//Abstract/AbstractText')
        pub_date_element = article.find('.//PubDate/Year')

        article_data['title'] = title_element.text if title_element is not None else 'No title available'
        article_data['abstract'] = abstract_element.text if abstract_element is not None else 'No abstract available'
        
        authors = []
        for author in article.findall('.//Author'):
            last_name = author.find('.//LastName')
            initials = author.find('.//Initials')
            if last_name is not None and initials is not None:
                authors.append(last_name.text + " " + initials.text)
            else:
                authors.append("Author details unavailable")
        article_data['authors'] = authors

        article_data['pub_date'] = pub_date_element.text if pub_date_element is not None else 'No publication date'

        articles.append(article_data)
    return articles

def get_papers_by_date(papers, start_year, end_year):
    papers_before_start = []
    papers_in_range = []
    for paper in papers:
        try:
            paper_year = int(paper['pub_date'])
            if paper_year < start_year:
                papers_before_start.append(paper)
            elif start_year <= paper_year <= end_year:
                papers_in_range.append(paper)
        except KeyError:
            print(f"Missing 'date' for paper: {paper['title']}")
        except ValueError:
            print(f"Invalid date format for paper: {paper['title']}")
    return papers_before_start, papers_in_range

@app.route('/fetch_articles', methods=['POST'])
def fetch_articles():
    data = request.json
    logging.debug('Received request:', data)  # Log the incoming request data
    try:
        query = data['query']
        mindate = 2000
        maxdate = 2020
        max_records = int(data['max_records'])  # Convert max_records to an integer
        article_ids = fetch_pubmed_data(query, mindate, maxdate, max_records)
        xml_data = fetch_details(article_ids)
        articles_metadata = parse_article_metadata(xml_data)
        logging.debug('Fetched articles metadata:', articles_metadata)  # Log the fetched metadata
        grouped_papers = get_papers_by_date(articles_metadata, int(data["mindate"]), int(data["maxdate"]))
        return jsonify(grouped_papers)
    except Exception as e:
        logging.error('Error in /fetch_articles route:', e)  # Log any errors
        return jsonify({'error': str(e)}), 500

def embed_text(text):
    """
    Embeds text by tokenizing and passing through a pre-trained model, returning the averaged last hidden state.
    """
    if text is None:
        text="..."
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

@app.route('/cluster_papers', methods=['POST'])
def cluster_papers():
    data = request.json
    papers_within_global = data['new_papers']
    papers_before_global = data['old_papers']
    
    def find_closest_papers(query_papers, all_papers, top_k=5, similarity_threshold=0.8):
        if not query_papers or not all_papers:
            return []

        query_vecs = []
        for paper in query_papers:
            embedded_text = embed_text(paper['abstract'])
            if embedded_text is not None:
                query_vecs.append(embedded_text)

        if not query_vecs:
            return []

        query_vecs = torch.stack(query_vecs)
        all_paper_vecs = []
        all_paper_texts = []
        for paper in all_papers:
            embedded_text = embed_text(paper['abstract'])
            if embedded_text is not None:
                all_paper_vecs.append(embedded_text)
                all_paper_texts.append(paper['abstract'])

        if not all_paper_vecs:
            return []

        all_paper_vecs = torch.stack(all_paper_vecs)
        similarities = cosine_similarity(query_vecs.numpy(), all_paper_vecs.numpy())
        best_assignments = {}

        for i, query_paper in enumerate(query_papers):
            closest_indices = similarities[i].argsort()[::-1]
            filtered_indices = [index for index in closest_indices if similarities[i][index] >= similarity_threshold]
            if len(filtered_indices) > top_k:
                filtered_indices = filtered_indices[:top_k]

            for index in filtered_indices:
                similarity_score = similarities[i][index]
                paper_title = all_papers[index]['title']
                if paper_title not in best_assignments or best_assignments[paper_title][1] < similarity_score:
                    best_assignments[paper_title] = (query_paper['title'], similarity_score)

        results = {query['title']: [] for query in query_papers}
        for paper_title, (query_title, _) in best_assignments.items():
            paper_abstract = next(p['abstract'] for p in all_papers if p['title'] == paper_title)
            similarity_score = best_assignments[paper_title][1]
            results[query_title].append((paper_title, paper_abstract, similarity_score))

        final_results = []
        for query_paper in query_papers:
            query_info = (query_paper['title'], query_paper['abstract'])
            closest_papers = sorted(results[query_paper['title']], key=lambda x: x[2], reverse=True)[:top_k]
            closest_papers = [paper for paper in closest_papers if paper[2] >= similarity_threshold]
            final_results.append((query_info, closest_papers))

        return final_results

    def fuse_similar_clusters(final_results, similarity_threshold=0.8):
        cluster_embeddings = {}
        for (query_title, query_abstract), papers in final_results:
            embeddings = [embed_text(paper[1]) for paper in papers]
            if embeddings:
                avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
                cluster_embeddings[query_title] = avg_embedding

        # Convert embeddings into a matrix for similarity comparison
        titles = list(cluster_embeddings.keys())
        embeddings_matrix = torch.stack(list(cluster_embeddings.values())).numpy()
        similarity_matrix = cosine_similarity(embeddings_matrix)

        # Identify clusters to merge based on the similarity threshold
        to_merge = {}
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                if similarity_matrix[i][j] > similarity_threshold:
                    # Assign smaller index cluster to merge into larger index cluster
                    smaller, larger = sorted([i, j], key=lambda x: -len(final_results[x][1]))
                    to_merge[titles[smaller]] = titles[larger]

        # Prepare to merge clusters and handle duplicates
        merged_clusters = {}
        for (query_title, query_abstract), papers in final_results:
            root_title = to_merge.get(query_title, query_title)
            if root_title not in merged_clusters:
                merged_clusters[root_title] = [(query_title, query_abstract), []]
            merged_clusters[root_title][1].extend(papers)

        # Convert merged clusters back to the expected format
        new_final_results = list(merged_clusters.values())

        for cluster in new_final_results:
            cluster[1].sort(key=lambda x: x[2], reverse=True)

        # Convert float32 values to float
        for cluster in new_final_results:
            cluster[1] = [(title, abstract, float(similarity)) for (title, abstract, similarity) in cluster[1]]

        return new_final_results

    def perform_pca(clusters):
        all_papers = []
        query_papers = []

        for cluster in clusters:
            query_papers.append(cluster[0])
            all_papers.extend(cluster[1])

        abstracts = [paper[1] for paper in all_papers]
        embeddings = [embed_text(abstract) for abstract in abstracts]
        embeddings_matrix = torch.stack(embeddings).numpy()

        pca = PCA(n_components=3)
        pca_results = pca.fit_transform(embeddings_matrix)

        pca_papers = []
        for idx, (title, abstract, similarity) in enumerate(all_papers):
            pca_papers.append({
                'title': title,
                'coords': pca_results[idx].tolist()
            })

        pca_query_papers = []
        for idx, (title, abstract) in enumerate(query_papers):
            pca_query_papers.append({
                'title': title,
                'coords': pca_results[idx].tolist()
            })

        return pca_query_papers, pca_papers

    # Find closest papers
    results = find_closest_papers(papers_within_global, papers_before_global, top_k=30, similarity_threshold=0.8)
    
    # Fuse similar clusters
    results = fuse_similar_clusters(results, similarity_threshold=0.8)
    
    # Perform PCA on the merged clusters
    pca_query_papers, pca_papers = perform_pca(results)

    return jsonify({'results': results, 'pca_results': {'query_papers': pca_query_papers, 'all_papers': pca_papers}})


@app.route('/visualize_clusters', methods=['POST'])
def visualize_clusters():
    data = request.json
    results_truncated = data['clusters']
    combined_papers = data['combined_papers']

    def generate_colors(num_colors):
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            lightness = 0.5
            saturation = 0.9
            color = mcolors.hsv_to_rgb((hue, saturation, lightness))
            colors.append(f'rgb({color[0]*255}, {color[1]*255}, {color[2]*255})')
        return colors

    all_embeddings = {}
    all_titles = []

    for query_title, similar_papers in results_truncated:
        query_abstract = next((p['abstract'] for p in combined_papers if p['title'] == query_title[0]), None)
        if query_abstract:
            all_embeddings[query_title[0]] = embed_text(query_abstract)
            all_titles.append(query_title[0])
        else:
            print(f"Abstract for query title '{query_title[0]}' not found.")

        for similar_title, _, _ in similar_papers:
            if similar_title not in all_embeddings:
                similar_abstract = next((p['abstract'] for p in combined_papers if p['title'] == similar_title), None)
                if similar_abstract:
                    all_embeddings[similar_title] = embed_text(similar_abstract)
                    all_titles.append(similar_title)
                else:
                    print(f"Abstract for similar title '{similar_title}' not found.")

    if not all_embeddings:
        return jsonify({'error': 'No embeddings available to perform PCA.'}), 500

    embeddings_matrix = torch.stack(list(all_embeddings.values())).numpy()
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings_matrix)
    paper_to_coords = dict(zip(all_titles, reduced_embeddings))

    traces = []
    num_queries = len(results_truncated)
    colors = generate_colors(num_queries)

    for idx, (query, similar_papers) in enumerate(results_truncated):
        query_coords = paper_to_coords.get(query[0], [0, 0, 0])
        query_trace = go.Scatter3d(
            x=[query_coords[0]], y=[query_coords[1]], z=[query_coords[2]],
            mode='markers',
            marker=dict(size=10, color=colors[idx]),
            name=f'Query {idx + 1}'
        )
        traces.append(query_trace)
        
        for title, _, _ in similar_papers:
            paper_coords = paper_to_coords.get(title, [0, 0, 0])
            paper_trace = go.Scatter3d(
                x=[paper_coords[0]], y=[paper_coords[1]], z=[paper_coords[2]],
                mode='markers',
                marker=dict(size=5, symbol='diamond', color=colors[idx]),
                showlegend=False
            )
            traces.append(paper_trace)

    layout = go.Layout(
        title="3D PCA Visualization of Paper Embeddings",
        scene=dict(
            xaxis=dict(title='PCA Component 1'),
            yaxis=dict(title='PCA Component 2'),
            zaxis=dict(title='PCA Component 3')
        ),
        legend=dict(
            title="Query Papers",
            itemsizing='constant',
            x=1.05,
            xanchor='left',
            orientation='v'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.show()
    return jsonify({'message': 'Visualization generated successfully.'})

@app.route('/generate_cluster_names', methods=['POST'])
def generate_cluster_names():
    data = request.json
    clusters = data['clusters']

    cluster_names = []
    for cluster in clusters:
        request_payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """Give an appropriate name for the given cluster."""
                },
                {
                    "role": "user",
                    "content": f"""Here is the data of new academic papers and their most similar but older papers.
                Given this information, please give a name for the cluster.
                Only give and say the name, no other tokens is needed from you. Only the name you found."""
                },
                {
                    "role": "user",
                    "content": json.dumps(cluster)
                }
            ]
        }

        req = requests.post(url, headers=auth_headers, json=request_payload, verify=False)
        res = req.json()
        cluster_name = f'"{res["choices"][0]["message"]["content"].strip()}"'
        cluster_names.append(cluster_name)

    return jsonify(cluster_names)

@app.route('/identify_trends', methods=['POST'])
def identify_trends():
    data = request.json
    clusters = data['clusters']

    trends = []
    for cluster in clusters:
        request_payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """Analyze the dataset consisting of new academic papers and their similar older counterparts. 
                    Identify trends, shifts in research focus, and notable thematic or methodological changes."""
                },
                {
                    "role": "user",
                    "content": f"""Here is the data of new academic papers and their most similar but older papers. 
                    Given this information, please analyze and extract significant trends, research focus shifts, and thematic discontinuities.
                    The response should be of this form:
                    - Trend 1: ... 
                    - Trend 2: ..."""
                },
                {
                    "role": "user",
                    "content": json.dumps(cluster)
                }
            ]
        }

        req = requests.post(url, headers=auth_headers, json=request_payload, verify=False)
        res = req.json()
        trends.append(res["choices"][0]["message"]["content"])

    return jsonify(trends)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
