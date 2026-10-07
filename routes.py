from flask import request, jsonify
import logging
from pubmed import fetch_pubmed_data, fetch_details, parse_article_metadata, get_papers_by_date
from clustering import find_closest_papers, fuse_similar_clusters, perform_pca
from ai import LLMNotConfigured, generate_cluster_name, identify_trends_in_cluster
from cache import get_from_cache, save_to_cache
from config import llm_configured, llm_provider, ollama_configured, ollama_model

global_query = None
global_max_records = None
global_presentdate = None
global_old_papers = None
global_new_papers = None
global_sim_tr = None
global_clusters = None
global_names = None


def initialize_routes(app):
    """
    Initialize routes for the Flask app and define a route to fetch articles.
    """

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "llm_configured": llm_configured(),
            "llm_provider": llm_provider() or None,
            "ollama_running": ollama_configured(),
            "ollama_model": ollama_model,
        })

    @app.route("/fetch_articles", methods=["POST"])
    @app.route("/fetch", methods=["POST"])
    def fetch_articles():
        global global_query, global_max_records, global_presentdate
        data = request.json or {}
        logging.info("Received fetch request: %s", {k: data.get(k) for k in ("query", "max_records", "presentdate")})
        try:
            query = data["query"]
            presentdate = int(data["presentdate"])
            max_records = int(data["max_records"])
            new_ids = fetch_pubmed_data(query, presentdate, presentdate, max_records)
            old_ids = fetch_pubmed_data(query, 2000, presentdate - 1, max_records)
            article_ids = list(dict.fromkeys(old_ids + new_ids))
            xml_data = fetch_details(article_ids)
            articles_metadata = parse_article_metadata(xml_data)
            logging.info("Fetched %s articles", len(articles_metadata))
            grouped_papers = get_papers_by_date(articles_metadata, presentdate)

            global_query = query
            global_max_records = max_records
            global_presentdate = presentdate

            old_papers, new_papers = grouped_papers
            return jsonify([old_papers, new_papers])
        except Exception as e:
            logging.exception("Error in /fetch_articles route")
            return jsonify({"error": str(e)}), 500

    @app.route("/cluster_papers", methods=["POST"])
    @app.route("/cluster", methods=["POST"])
    def cluster_papers():
        global global_query, global_max_records, global_presentdate, global_old_papers, global_new_papers, global_sim_tr
        global global_clusters
        data = request.json or {}
        query = global_query
        max_papers = global_max_records
        present_year = global_presentdate
        new_papers = data["new_papers"]
        old_papers = data["old_papers"]
        global_new_papers = new_papers
        global_old_papers = old_papers
        sim_threshold = float(data["sim_threshold"])
        global_sim_tr = sim_threshold

        try:
            cached_result = get_from_cache(query, max_papers, present_year, new_papers, old_papers, sim_threshold)
            if cached_result and cached_result.get("clusters"):
                results = cached_result["clusters"]
            else:
                results = find_closest_papers(new_papers, old_papers)
                results = fuse_similar_clusters(results, similarity_threshold=sim_threshold)
                save_to_cache(query, max_papers, present_year, new_papers, old_papers, sim_threshold, results, None, None)

            global_clusters = results
            pca_query_papers, pca_papers = perform_pca(results)
            return jsonify({
                "results": results,
                "pca_results": {"query_papers": pca_query_papers, "all_papers": pca_papers},
            })
        except Exception as e:
            logging.exception("Error in /cluster_papers route")
            return jsonify({"error": str(e)}), 500

    @app.route("/visualize_clusters", methods=["POST"])
    def visualize_clusters():
        return jsonify({
            "message": "3D visualization is rendered in the browser from /cluster_papers PCA results."
        })

    @app.route("/generate_cluster_names", methods=["POST"])
    def generate_cluster_names():
        global global_query, global_max_records, global_presentdate, global_old_papers, global_new_papers, global_sim_tr
        global global_clusters, global_names
        data = request.json or {}
        clusters = data.get("clusters") or global_clusters or []
        cluster_names = []

        try:
            cached_result = get_from_cache(
                global_query, global_max_records, global_presentdate,
                global_new_papers, global_old_papers, global_sim_tr,
            )
            if cached_result and cached_result.get("cluster_names") is not None:
                cluster_names = cached_result["cluster_names"]
            else:
                for cluster in clusters:
                    cluster_name = generate_cluster_name(cluster)
                    cluster_names.append(cluster_name)
                save_to_cache(
                    global_query, global_max_records, global_presentdate,
                    global_new_papers, global_old_papers, global_sim_tr,
                    global_clusters, cluster_names, None,
                )

            global_names = cluster_names
            return jsonify(cluster_names)
        except LLMNotConfigured as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            logging.exception("Error in /generate_cluster_names route")
            return jsonify({"error": str(e)}), 500

    @app.route("/identify_trends", methods=["POST"])
    def identify_trends():
        global global_query, global_max_records, global_presentdate, global_old_papers, global_new_papers, global_sim_tr
        global global_clusters, global_names
        data = request.json or {}
        clusters = data.get("clusters") or global_clusters or []
        trends = []

        try:
            cached_result = get_from_cache(
                global_query, global_max_records, global_presentdate,
                global_new_papers, global_old_papers, global_sim_tr,
            )
            if cached_result and cached_result.get("trends") is not None:
                trends = cached_result["trends"]
            else:
                for cluster in clusters:
                    trend = identify_trends_in_cluster(cluster)
                    trends.append(trend)
                save_to_cache(
                    global_query, global_max_records, global_presentdate,
                    global_new_papers, global_old_papers, global_sim_tr,
                    global_clusters, global_names, trends,
                )

            return jsonify(trends)
        except LLMNotConfigured as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            logging.exception("Error in /identify_trends route")
            return jsonify({"error": str(e)}), 500
