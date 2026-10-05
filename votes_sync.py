"""
Download new votes from S3 and append them to a local CSV.

The CSV is the record of what has already been downloaded: every run lists the
bucket, skips vote_ids already in the CSV, and appends only the new votes.
No per-vote JSON files are kept locally.

    python votes_sync.py              # from the terminal
    from votes_sync import sync_votes # from a notebook
"""

import csv
import json
import re
from pathlib import Path

import boto3

BUCKET = "dinogpt-votes"
PREFIX = "votes/"
PROFILE = "dinogpt-trainer"             # the read-only key
CSV_PATH = Path("votes/votes.csv")      # votes/ is git-ignored

# Column order in the CSV: the fields of a vote, as the app writes them.
FIELDS = ["vote_id", "timestamp", "model", "seed", "name_1", "name_2",
          "preference", "schema_version"]

PREFERENCES = {"name_1", "name_2", "both_fine", "both_bad"}
LETTERS = re.compile(r"[a-z]*")
VOTE_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def is_valid_vote(vote):
    # Same rules the app applies before saving. Anything in the bucket that
    # doesn't match (a test file, or junk from a leaked key) is skipped.
    try:
        return (
            set(FIELDS) <= vote.keys()
            and VOTE_ID.fullmatch(vote["vote_id"]) is not None
            and vote["preference"] in PREFERENCES
            and all(LETTERS.fullmatch(vote[k]) for k in ("seed", "name_1", "name_2"))
            and len(vote["seed"]) <= 20
            and all(len(vote[k]) <= 27 for k in ("name_1", "name_2"))
            and vote["name_1"] != vote["name_2"]
        )
    except (TypeError, AttributeError):
        return False


def is_empty(csv_path):
    return not csv_path.exists() or csv_path.stat().st_size == 0


def known_vote_ids(csv_path):
    # vote_ids already in the CSV: always the first column. An empty set if the
    # CSV doesn't exist yet or is empty. Copes with a file that has no header row.
    if is_empty(csv_path):
        return set()
    with csv_path.open(newline="") as f:
        return {row[0] for row in csv.reader(f) if row and row[0] != "vote_id"}


def list_vote_keys(s3, bucket=BUCKET):
    # Every vote file in the bucket. One list call returns at most 1,000 keys,
    # so the paginator fetches the rest.
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip the "votes/" folder marker the console creates, and anything else
            # that isn't a vote file.
            if key.endswith(".json"):
                yield key


def sync_votes(csv_path=CSV_PATH, bucket=BUCKET, profile=PROFILE, s3=None):
    """Append votes not yet in csv_path. Returns the number of new votes added."""
    csv_path = Path(csv_path)
    s3 = s3 or boto3.Session(profile_name=profile).client("s3")

    known = known_vote_ids(csv_path)
    new_votes, skipped = [], []

    for key in list_vote_keys(s3, bucket):
        vote_id = Path(key).stem                    # votes/<vote_id>.json -> <vote_id>
        if vote_id in known:
            continue

        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        try:
            vote = json.loads(body)
        except json.JSONDecodeError:
            skipped.append(key)
            continue

        if not is_valid_vote(vote) or vote["vote_id"] != vote_id:
            skipped.append(key)
            continue

        new_votes.append(vote)
        known.add(vote_id)

    # Oldest first, so the CSV reads in voting order.
    new_votes.sort(key=lambda v: v["timestamp"])

    if new_votes:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = is_empty(csv_path)      # new file, or an empty one
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(new_votes)

    print(f"{len(new_votes)} new votes added to {csv_path} ({len(known)} in total).")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) that aren't valid votes: {skipped}")
    return len(new_votes)


if __name__ == "__main__":
    sync_votes()