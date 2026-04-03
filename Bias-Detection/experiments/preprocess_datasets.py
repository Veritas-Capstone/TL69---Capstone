"""
Unified preprocessing pipeline for all 12 political leaning datasets.
Based on the notebooks from the volf model github page
"""

import csv
import gc
import json
import os
import sqlite3
from io import StringIO
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_dataset

import sys
sys.path.append(str(Path(__file__).parent))
from dataset_utils import PoliticalLeaningDataset

# which datasets to process (set to a list of names, or "all")
DATASETS_TO_PROCESS = "all"


class DatasetPreprocessor:

    def __init__(self, name):
        self.name = name
        self.raw_dir = Path("raw")
        self.preprocessed_dir = Path("preprocessed")
        self.preprocessed_dir.mkdir(exist_ok=True)

    def add_politicalness_label(self, df):
        df["politicalness"] = pd.Categorical(["political"] * len(df))
        return df

    def strip_and_clean(self, df):
        if "title" in df.columns:
            df["title"] = (df["title"].str.strip()).replace("", np.nan)
        if "body" in df.columns:
            df["body"] = (df["body"].str.strip()).replace("", np.nan)
        return df

    def drop_missing(self, df):
        df = df.dropna(subset=["leaning"])
        if "title" in df.columns:
            df = df.dropna(subset=["title", "body"], how="all")
        else:
            df = df.dropna(subset=["body"])
        return df

    def remove_duplicates(self, df):
        return df.drop_duplicates(subset="body")

    def add_length_stats(self, df):
        if "title" in df.columns:
            df["title_length"] = df["title"].fillna("").str.len()
            df["title_word_count"] = df["title"].fillna("").str.split().str.len()
        df["body_length"] = df["body"].fillna("").str.len()
        df["body_word_count"] = df["body"].fillna("").str.split().str.len()
        return df

    def save(self, df):
        output_path = self.preprocessed_dir / f"{self.name}.parquet"
        df.to_parquet(output_path)
        print(f"  Saved {len(df)} rows to {output_path}")
        return df


class ArticleBiasPredictionPreprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        raw_dir = self.raw_dir / "article_bias_prediction" / "data" / "jsons"
        if not raw_dir.exists():
            raw_dir = self.raw_dir / "article_bias_prediction"

        json_files = [f for f in os.listdir(raw_dir) if f.endswith('.json')]
        print(f"  Found {len(json_files)} JSON files")

        data = []
        for idx, filename in enumerate(json_files, 1):
            with open(raw_dir / filename, "r", encoding="utf-8") as file:
                data.append(json.load(file))
            if idx % 500 == 0:
                print(f"  Loading... {idx}/{len(json_files)}")

        df = pd.DataFrame(data)
        df = df.drop(columns=["topic", "bias", "source", "url", "date", "authors",
                               "content_original", "source_url", "ID"], errors='ignore')
        df = df.rename(columns={"content": "body", "bias_text": "leaning"})
        df = df[["title", "body", "leaning"]].copy()
        df["leaning"] = df["leaning"].astype("category")
        df = self.add_politicalness_label(df)
        df = self.strip_and_clean(df)
        df = self.drop_missing(df)
        df = self.remove_duplicates(df)
        df = self.add_length_stats(df)
        df = df[df["body_word_count"] >= 40]
        return self.save(df)


class BIGNEWSBLNPreprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        bignews_dir = self.raw_dir / "bignewsbln"

        df_left = pd.read_json(open(bignews_dir / "BIGNEWSBLN_left.json", encoding="utf-8"))
        df_center = pd.read_json(open(bignews_dir / "BIGNEWSBLN_center.json", encoding="utf-8"))
        df_right = pd.read_json(open(bignews_dir / "BIGNEWSBLN_right.json", encoding="utf-8"))

        df_left["leaning"] = ["left"] * len(df_left)
        df_center["leaning"] = ["center"] * len(df_center)
        df_right["leaning"] = ["right"] * len(df_right)

        df = pd.concat([df_left, df_center, df_right])
        del df_left, df_center, df_right
        gc.collect()

        df["leaning"] = df["leaning"].astype("category")
        df = self.add_politicalness_label(df)
        df = df.drop(columns=["date", "url", "source", "html"], errors='ignore')
        df = df.rename(columns={"text": "body"})
        df = df[["title", "body", "leaning", "politicalness"]].copy()

        if df["body"].dtype == object and isinstance(df["body"].iloc[0], list):
            df["body"] = df["body"].apply(lambda parts: " ".join(parts))

        df = self.strip_and_clean(df)
        df = self.drop_missing(df)
        df = self.remove_duplicates(df)
        df = self.add_length_stats(df)

        dataset = PoliticalLeaningDataset(self.name, df)
        df = dataset.take_even_class_sample_by_size(100_000).dataframe
        return self.save(df)


class CommonCrawlNewsArticlesPreprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        db_path = self.raw_dir / "commoncrawl_news_articles" / "articles.db"
        if not db_path.exists():
            db_path = self.raw_dir / "commoncrawl_news_articles" / "news_articles.db"

        connection = sqlite3.connect(str(db_path))
        df = pd.read_sql(
            "SELECT content_preprocessed, au.outlet_name FROM article_contents ac "
            "INNER JOIN article_urls au on ac.uuid = au.uuid WHERE language = 'en'",
            connection
        )

        outlets_path = self.raw_dir / "commoncrawl_news_articles" / "outlets.json"
        if not outlets_path.exists():
            outlets_path = self.raw_dir / "commoncrawl_news_articles" / "domain-outlet-map.json"

        outlets_df = pd.read_json(open(outlets_path, encoding="utf-8"))
        df = df.merge(outlets_df, how="outer", left_on="outlet_name", right_on="name")

        df = df.drop(columns=["outlet_name", "name", "tld", "filter", "allsides_name"], errors='ignore')
        df = df.rename(columns={"content_preprocessed": "body", "allsides_rating": "leaning"})
        df["leaning"] = df["leaning"].replace({"lean left": "left", "lean right": "right"})
        df["leaning"] = df["leaning"].astype("category")
        df = self.add_politicalness_label(df)
        df = self.strip_and_clean(df)
        df = df.dropna()
        df = self.remove_duplicates(df)
        df["body"] = df["body"].str.replace("<SENT_END>", " ")
        df = self.add_length_stats(df)
        df = df[df["body_word_count"] >= 50]

        dataset = PoliticalLeaningDataset(self.name, df)
        df = dataset.take_even_class_sample_by_size(100_000).dataframe
        return self.save(df)


class DemRepPartyPlatformTopicsPreprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        ds = load_dataset("mlburnham/dem_rep_party_platform_topics",
                          revision="08af680e8575e47c4f49e69bc458c5a423b8376b")
        df = concatenate_datasets(ds.values()).to_pandas()
        df = df.drop(columns=["target", "hypothesis", "entailment",
                               "validation_label", "validation_source"], errors='ignore')
        df = df.rename(columns={"premise": "body", "dataset": "leaning"})
        df["leaning"] = df["leaning"].replace({
            "Dem. Party Platforms": "left", "Rep. Party Platforms": "right"
        })
        df["leaning"] = df["leaning"].astype("category")
        df = self.add_politicalness_label(df)
        df = self.strip_and_clean(df)
        df = df.dropna()
        df = self.remove_duplicates(df)
        df = self.add_length_stats(df)
        return self.save(df)


class GPT4PoliticalBiasPreprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        ds = load_dataset("cajcodes/political-bias",
                          revision="f24cd353b9c4a69c274fab4e43610ad90b1ae0d2")
        df = concatenate_datasets(ds.values()).to_pandas()
        df = df.rename(columns={"text": "body", "label": "leaning"})
        # 5-level to 3-level
        df["leaning"] = df["leaning"].replace({0: "right", 1: "right", 2: "center", 3: "left", 4: "left"})
        df["leaning"] = df["leaning"].astype("category")
        df = self.add_politicalness_label(df)
        df = self.strip_and_clean(df)
        df = df.dropna()
        df = self.remove_duplicates(df)
        df = self.add_length_stats(df)
        return self.save(df)


class GPT4PoliticalIdeologiesPreprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        ds = load_dataset("JyotiNayak/political_ideologies",
                          revision="f748ec8a7cbbf916453ba489fdc9766b9e4f19c8")
        df = concatenate_datasets(ds.values()).to_pandas()
        df = df.set_index("__index_level_0__").rename_axis(None)
        df = df.drop(columns=["issue_type"], errors='ignore')
        df = df.rename(columns={"statement": "body", "label": "leaning"})
        df["leaning"] = df["leaning"].replace({0: "right", 1: "left"})
        df["leaning"] = df["leaning"].astype("category")
        df = self.add_politicalness_label(df)
        df = self.strip_and_clean(df)
        df = df.dropna()
        df = self.add_length_stats(df)
        return self.save(df)


class MediaPoliticalStancePreprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        tsv_path = self.raw_dir / "media_political_stance.tsv"
        if not tsv_path.exists():
            tsv_path = self.raw_dir / "media_political_stance" / "media_political_stance.tsv"

        df = pd.read_csv(tsv_path, sep="\t", header=None,
                         names=["topic10", "topic15", "stance", "oscarID", "url", "text"],
                         encoding="utf-8")
        df = df.drop(columns=["topic10", "topic15", "oscarID", "url"], errors='ignore')
        df = df.rename(columns={"text": "body", "stance": "leaning"})
        df = df[["body", "leaning"]].copy()
        df["leaning"] = df["leaning"].astype("category")
        df = self.add_politicalness_label(df)
        df = self.strip_and_clean(df)
        df = df.dropna()
        df = self.remove_duplicates(df)
        df["body"] = df["body"].str.replace(" <NS>", "")
        df = self.add_length_stats(df)

        dataset = PoliticalLeaningDataset(self.name, df)
        df = dataset.take_even_class_sample_by_size(100_000).dataframe
        return self.save(df)


class PoliticalPodcastsPreprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        path = kagglehub.dataset_download(
            "nbandhi/political-podcasts-listing-with-audio-links/versions/1")
        df = pd.read_csv(Path(path) / "politicalpodcasts.csv", encoding="utf-8")
        df = df.set_index("Unnamed: 0").rename_axis(None)
        df = df.drop(columns=["podcaster", "pub_date", "pod_link"], errors='ignore')
        df = df.rename(columns={"abstract": "body", "type": "leaning"})
        df = df[["title", "body", "leaning"]].copy()
        df["leaning"] = df["leaning"].replace({"liberal": "left", "conservative": "right"})
        df["leaning"] = df["leaning"].astype("category")
        df = self.add_politicalness_label(df)
        df = self.strip_and_clean(df)
        df = df.dropna()
        df = self.remove_duplicates(df)
        df = self.add_length_stats(df)
        df = df[df["body_word_count"] >= 6]
        return self.save(df)


class PoliticalTweetsPreprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        ds = load_dataset("Jacobvs/PoliticalTweets",
                          revision="1ddaa14beed79edda621fdd72ad22fd654d760b3")
        df = concatenate_datasets(ds.values()).to_pandas()
        df = df.set_index("index").rename_axis(None)
        df = df.drop(columns=["id", "username", "labels"], errors='ignore')
        df = df.rename(columns={"text": "body", "party": "leaning"})
        df["leaning"] = df["leaning"].replace({"Democrat": "left", "Republican": "right"})
        df["leaning"] = df["leaning"].astype("category")
        df = self.add_politicalness_label(df)
        df = self.strip_and_clean(df)
        df = df.dropna()
        df = self.remove_duplicates(df)
        df = self.add_length_stats(df)
        df = df[df["body_word_count"] >= 2]

        dataset = PoliticalLeaningDataset(self.name, df)
        df = dataset.take_even_class_sample_by_size(100_000).dataframe
        return self.save(df)


class QbiasPreprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        qbias_path = self.raw_dir / "qbias" / "qbias.csv"
        if not qbias_path.exists():
            qbias_path = self.raw_dir / "qbias.csv"

        df = pd.read_csv(qbias_path, encoding="utf-8")
        df = df.drop(columns=["Unnamed: 0", "tags", "heading", "source"], errors='ignore')
        df = df.rename(columns={"text": "body", "bias_rating": "leaning"})
        df["leaning"] = df["leaning"].astype("category")
        df = self.add_politicalness_label(df)
        df = self.strip_and_clean(df)
        df = self.drop_missing(df)
        df = self.remove_duplicates(df)
        df = self.add_length_stats(df)
        df = df[df["body_word_count"] >= 4]
        return self.save(df)


class WebisBiasFlipper18Preprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        csv.field_size_limit(4 * 131072)

        csv_path = self.raw_dir / "webis_bias_flipper_18.csv"
        if not csv_path.exists():
            csv_path = self.raw_dir / "webis_bias_flipper_18" / "corpus.csv"
        if not csv_path.exists():
            csv_path = self.raw_dir / "webis_bias_flipper_18" / "data_public.csv"

        with open(csv_path, mode="r", encoding="utf-8") as file:
            modified_lines = (
                "\u2603".join(
                    line.replace('","', "\u2603,\u2603")
                    .replace('"', "\u2603", 1)
                    .rsplit('"', 1)
                )
                for line in file
            )
            df = pd.read_csv(StringIO("".join(modified_lines)),
                             quotechar="\u2603", encoding="utf-8", engine="python")

        df = df.drop(columns=["story_id", "title", "body", "source"], errors='ignore')
        df = df.rename(columns={"original_title": "title", "original_body": "body", "bias": "leaning"})
        df = df[["title", "body", "leaning"]].copy()
        df["leaning"] = df["leaning"].astype("category")
        df["leaning"] = df["leaning"].cat.rename_categories({
            "From the Left": "left", "From the Center": "center", "From the Right": "right"
        })
        df = self.add_politicalness_label(df)
        df = self.strip_and_clean(df)
        df = self.drop_missing(df)
        df = self.remove_duplicates(df)
        df = self.add_length_stats(df)
        df = df[df["body_word_count"] >= 15]
        return self.save(df)


class WebisNewsBias20Preprocessor(DatasetPreprocessor):
    def preprocess(self):
        print(f"\nProcessing {self.name}...")
        json_path = self.raw_dir / "webis_news_bias_20.json"
        if not json_path.exists():
            json_path = self.raw_dir / "webis_news_bias_20" / "corpus.jsonl"
        if not json_path.exists():
            json_path = self.raw_dir / "webis_news_bias_20.jsonl"

        df = pd.read_json(open(json_path, encoding="utf-8"), lines=True)
        df = df.drop(columns=["source", "event_id", "adfontes_fair",
                               "adfontes_political", "misc"], errors='ignore')
        df = df.rename(columns={"content": "body", "allsides_bias": "leaning"})
        df = df[["title", "body", "leaning"]].copy()
        df["leaning"] = df["leaning"].astype("category")
        df["leaning"] = df["leaning"].cat.rename_categories({
            "From the Left": "left", "From the Center": "center", "From the Right": "right"
        })
        df = self.add_politicalness_label(df)
        df = self.strip_and_clean(df)
        df = self.drop_missing(df)
        df = self.remove_duplicates(df)
        df = self.add_length_stats(df)
        df = df[df["body_word_count"] >= 30]
        return self.save(df)


# dataset registry
PREPROCESSORS = {
    "article_bias_prediction": ArticleBiasPredictionPreprocessor,
    "bignewsbln": BIGNEWSBLNPreprocessor,
    "commoncrawl_news_articles": CommonCrawlNewsArticlesPreprocessor,
    "dem_rep_party_platform_topics": DemRepPartyPlatformTopicsPreprocessor,
    "gpt4_political_bias": GPT4PoliticalBiasPreprocessor,
    "gpt4_political_ideologies": GPT4PoliticalIdeologiesPreprocessor,
    "media_political_stance": MediaPoliticalStancePreprocessor,
    "political_podcasts": PoliticalPodcastsPreprocessor,
    "political_tweets": PoliticalTweetsPreprocessor,
    "qbias": QbiasPreprocessor,
    "webis_bias_flipper_18": WebisBiasFlipper18Preprocessor,
    "webis_news_bias_20": WebisNewsBias20Preprocessor,
}


if __name__ == "__main__":
    if DATASETS_TO_PROCESS == "all":
        datasets = PREPROCESSORS.keys()
    else:
        datasets = DATASETS_TO_PROCESS

    print("Political Leaning Datasets Preprocessing\n")

    results = {}
    for name in datasets:
        try:
            preprocessor = PREPROCESSORS[name](name)
            df = preprocessor.preprocess()
            results[name] = {"status": "success", "rows": len(df)}
        except Exception as e:
            print(f"  Error processing {name}: {e}")
            results[name] = {"status": "failed", "error": str(e)}

    print("\nSummary:")
    for name, result in results.items():
        if result["status"] == "success":
            print(f"  {name}: {result['rows']} rows")
        else:
            print(f"  {name}: FAILED - {result['error']}")