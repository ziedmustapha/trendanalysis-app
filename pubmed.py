import requests
import xml.etree.ElementTree as ET
import logging
import re

PUBMED_HEADERS = {
    "User-Agent": "trendanalysis-app/1.0 (research; PubMed E-utilities)",
}

DOCTYPE_RE = re.compile(r"<!DOCTYPE[^>]*>", re.IGNORECASE)
XML_DECL_RE = re.compile(r"<\?xml[^?]*\?>", re.IGNORECASE)


def fetch_pubmed_data(query, mindate, maxdate, max_records):
    """
    Fetch PubMed data based on the given query, date range, and maximum number of records.
    @param query - The search query for PubMed data.
    @param mindate - The minimum date for the search.
    @param maxdate - The maximum date for the search.
    @param max_records - The maximum number of records to fetch.
    @return A list of PubMed IDs that match the query and date range.
    """
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
        response = requests.get(base_url, params=params, headers=PUBMED_HEADERS, timeout=60)
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
    """
    Fetch details for a list of PubMed IDs using the NCBI E-utilities API.
    @param pubmed_ids - List of PubMed IDs to fetch details for
    @return Combined XML containing details for all PubMed IDs
    """
    if not pubmed_ids:
        return "<root></root>"
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
        response = requests.get(url, params=params, headers=PUBMED_HEADERS, timeout=60)
        response.raise_for_status()
        batch_xml = XML_DECL_RE.sub("", response.text)
        batch_xml = DOCTYPE_RE.sub("", batch_xml).strip()
        all_details.append(batch_xml)

    combined_xml = "<root>" + "".join(all_details) + "</root>"
    return combined_xml

def parse_article_metadata(xml_data):
    """
    Parse the metadata of an article from XML data.
    @param xml_data - the XML data containing the article metadata
    @return A list of dictionaries containing the parsed metadata of each article
    """
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        logging.error(f"Error parsing XML: {e}")
        return []

    articles = []
    for article in root.findall('.//PubmedArticle'):
        article_data = {}
        title_element = article.find('.//ArticleTitle')
        abstract_element = article.find('.//Abstract')
        pub_date_element = article.find('.//PubDate/Year')
        if pub_date_element is None:
            pub_date_element = article.find('.//ArticleDate/Year')

        article_data['title'] = ''.join(title_element.itertext()).strip() if title_element is not None else 'No title available'
        if abstract_element is not None:
            article_data['abstract'] = ' '.join(abstract_element.itertext()).strip() or 'No abstract available'
        else:
            article_data['abstract'] = 'No abstract available'
        
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

def get_papers_by_date(papers, presentdate):
    """
    Filter a list of papers based on their publication date compared to a given present date.
    @param papers - List of papers with publication dates.
    @param presentdate - The present date for filtering.
    @return Two lists: papers_before_start (papers published before presentdate) and papers_in_range (papers published in the presentdate)
    """
    papers_before_start = []
    papers_in_range = []
    for paper in papers:
        try:
            paper_year = int(paper['pub_date'])
            if paper_year < presentdate:
                papers_before_start.append(paper)
            elif paper_year == presentdate:
                papers_in_range.append(paper)
        except KeyError:
            print(f"Missing 'date' for paper: {paper['title']}")
        except ValueError:
            print(f"Invalid date format for paper: {paper['title']}")
    return papers_before_start, papers_in_range