import os
import requests
import shutil
import pandas as pd
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def download_data():
    source = os.environ["CJPTABLES"]
    destination_folder = "."
    gz_fn = os.path.join(destination_folder, "cjp_tables.tar.gz")
    if os.path.exists(os.path.join(destination_folder, "cjp_tables")):
        shutil.rmtree(os.path.join(destination_folder, "cjp_tables"))
    response = requests.get(source, stream=True)
    with open(gz_fn, "wb") as f:
        f.write(response.raw.read())
    shutil.unpack_archive(gz_fn, destination_folder)
    os.remove(gz_fn)


def load_all_data():
    # fmt: off
    """ articles_df, categories_df, trainedcategoryrelevance_df, trainedcoding_df, usercoding_df, usercoding_categories_df, trainedlocation"""
    # fmt: on
    newssource = load_newssource()
    articles = load_articles()
    categories = load_categories()
    trainedcategoryrelevance = load_trainedcategoryrelevance()
    trainedlocation = load_trainedlocation()
    trainedcoding = load_trainedcoding()
    usercoding = load_usercoding()
    usercoding_categories = load_usercoding_categories()
    trainedsentiment = load_trainedsentiment()
    trainedsentimententities = load_trainedsentimententities()
    return (
        newssource,
        articles,
        categories,
        trainedcategoryrelevance,
        trainedcoding,
        usercoding,
        usercoding_categories,
        trainedlocation,
        trainedsentiment,
        trainedsentimententities
    )


def load_newssource():
    newsource = pd.read_csv(
        "./cjp_tables/newsarticles_newssource.csv.gz", header=None, compression="gzip"
    )
    newsource.columns = [
        "source_id",
        "source_name",
        "short_name",
        "legacy_feed_id",
    ]
    print(f"news sources loaded. size: {newsource.shape}")
    return newsource


def load_articles():
    article = pd.read_csv(
        "./cjp_tables/newsarticles_article.csv.gz", header=None, compression="gzip"
    )
    article.columns = [
        "id",
        "feedname",
        "url",
        "orig_html",
        "title",
        "bodytext",
        "relevant",
        "created",
        "last_modified",
        "news_source_id",
        "author",
    ]
    print(f"articles loaded. size: {article.shape}")
    return article


def load_categories():
    categories = pd.read_csv(
        "./cjp_tables/newsarticles_category.csv.gz", header=None, compression="gzip"
    )
    categories.columns = ["id", "title", "abbreviation", "created", "active", "kind"]
    print(f"categories loaded. size: {categories.shape}")
    return categories


def load_trainedcategoryrelevance():
    trainedcategoryrelevance = pd.read_csv(
        "./cjp_tables/newsarticles_trainedcategoryrelevance.csv.gz", compression="gzip"
    )
    trainedcategoryrelevance.columns = ["id", "relevance", "category_id", "coding_id"]
    print(f"trainedcategoryrelevance loaded. size: {trainedcategoryrelevance.shape}")
    return trainedcategoryrelevance


def load_trainedcoding():
    trainedcoding = pd.read_csv(
        "./cjp_tables/newsarticles_trainedcoding.csv.gz",
        header=None,
        compression="gzip",
    )
    trainedcoding.columns = [
        "id",
        "date",
        "model_info",
        "relevance",
        "article_id",
        "sentiment",
        "bin",
        "sentiment_processed",
    ]
    print(f"trainedcoding loaded. size: {trainedcoding.shape}")
    return trainedcoding


def load_trainedlocation():
    trainedlocation = pd.read_csv(
        "./cjp_tables/newsarticles_trainedlocation.csv.gz",
        header=None,
        compression="gzip",
    )
    trainedlocation.columns = [
        "id",
        "text",
        "latitude",
        "longitude",
        "coding_id",
        "confidence",
        "neighborhood",
        "is_best"
    ]
    print(f"trainedlocation loaded. size: {trainedlocation.shape}")
    return trainedlocation


def load_usercoding():
    usercoding = pd.read_csv(
        "./cjp_tables/newsarticles_usercoding.csv.gz", header=None, compression="gzip"
    )
    usercoding.columns = [
        "id",
        "date",
        "relevant",
        "article_id",
        "user_id",
        "locations",
        "sentiment",
    ]
    print(f"usercoding loaded. size: {usercoding.shape}")
    return usercoding


def load_usercoding_categories():
    usercoding_categories = pd.read_csv(
        "./cjp_tables/newsarticles_usercoding_categories.csv.gz",
        header=None,
        compression="gzip",
    )
    usercoding_categories.columns = ["id", "usercoding_id", "category_id"]
    print(f"usercoding_categories loaded. size: {usercoding_categories.shape}")
    return usercoding_categories


def load_trainedsentiment():
    trainedsentiment = pd.read_csv(
        "./cjp/newsarticles_trainedsentiment.csv.gz",
        header=None,
        compression="gzip"
    )
    trainedsentiment.columns = [
        "id",
        "date",
        "api_response",
        "coding_id",
    ]
    print(f"trainedsentiment loaded. size: {trainedsentiment.shape}")
    return trainedsentiment


def load_trainedsentimententities():
    trainedsentimententities = pd.read_csv(
        "./cjp_tables/newsarticles_trainedsentimententities.csv.gz",
        header=None,
        compression="gzip"
    )
    trainedsentimententities.columns = [
        "id",
        "index",
        "entity",
        "sentiment",
        "coding_id",
        "response_id",
    ]
    print(f"trainedsentimententities loaded. size: {trainedsentimententities.shape}")
    return trainedsentimententities