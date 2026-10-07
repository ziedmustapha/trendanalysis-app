import sqlite3
import json

def create_cache():
    """
    Create a cache database to store information related to paper queries.
    @return None
    """
    conn = sqlite3.connect('cache.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            query TEXT,
            max_papers INTEGER,
            present_year INTEGER,
            new_papers TEXT,
            old_papers TEXT,
            sim_threshold REAL,
            clusters TEXT,
            cluster_names TEXT,
            trends TEXT,
            PRIMARY KEY (query, max_papers, present_year, new_papers, old_papers, sim_threshold)
        )
    ''')
    conn.commit()
    conn.close()

def get_from_cache(query, max_papers, present_year, new_papers, old_papers, sim_threshold):
    """
    Retrieve data from the cache database based on the provided query parameters.
    @param query - the query to retrieve data for
    @param max_papers - the maximum number of papers
    @param present_year - the present year
    @param new_papers - list of new papers
    @param old_papers - list of old papers
    @param sim_threshold - the similarity threshold
    @return A dictionary containing clusters, cluster names, and trends if found in the cache database, otherwise None.
    """
    conn = sqlite3.connect('cache.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT clusters, cluster_names, trends FROM cache
        WHERE query = ? AND max_papers = ? AND present_year = ? AND new_papers = ? AND old_papers = ? AND sim_threshold = ?
    ''', (query, max_papers, present_year, json.dumps(new_papers), json.dumps(old_papers), sim_threshold))
    row = cursor.fetchone()
    conn.close()
    if row:
        def _loads(value):
            if value is None:
                return None
            return json.loads(value)

        return {
            'clusters': _loads(row[0]),
            'cluster_names': _loads(row[1]),
            'trends': _loads(row[2])
        }
    return None

def save_to_cache(query, max_papers, present_year, new_papers, old_papers, sim_threshold, clusters, cluster_names, trends):
    """
    Save the query results and related information to a cache database.
    @param query - the query used to retrieve the data
    @param max_papers - the maximum number of papers to retrieve
    @param present_year - the current year
    @param new_papers - newly retrieved papers
    @param old_papers - previously retrieved papers
    @param sim_threshold - similarity threshold
    @param clusters - clusters of papers
    @param cluster_names - names of the clusters
    @param trends - trends in the data
    """
    conn = sqlite3.connect('cache.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO cache (query, max_papers, present_year, new_papers, old_papers, sim_threshold, clusters, cluster_names, trends)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (query, max_papers, present_year, json.dumps(new_papers), json.dumps(old_papers), sim_threshold, json.dumps(clusters), json.dumps(cluster_names), json.dumps(trends)))
    conn.commit()
    conn.close()
