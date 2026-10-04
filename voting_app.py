import json
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import streamlit as st
import torch

from dinohelper import Decoder, generate_name

MODEL_FILE = "dino_gpt_original.pth"
VOTES_DIR = Path(__file__).parent / "votes"
MAX_SEED_LEN = 20   # leaves room to generate within the 27-character window
MAX_INPUT_LEN = 50  # hard cap on what the text box accepts at all
MAX_TRIES = 5       # attempts to get two different names
PREFERENCES = {"name_1", "name_2", "both_fine", "both_bad"}
LETTERS = re.compile(r"[a-z]*")


# ---------------------------------------------------------------
# Model: loaded once per server, not on every rerun
# ---------------------------------------------------------------

@st.cache_resource
def load_model(path):
    model = Decoder()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


model = load_model(MODEL_FILE)


# ---------------------------------------------------------------
# Generating a pair
# ---------------------------------------------------------------

def clean_seed(text):
    # Reduce anything typed to plain a-z, the only characters the model knows:
    #   "Ankylosaurüs"  -> "ankylosaurus"  (accents dropped, letter kept)
    #   "T-Rex 2!"      -> "trex"          (everything else removed)
    #   "<script>"      -> "script"        (no markup survives)
    text = unicodedata.normalize("NFKD", text)          # ü -> u + accent mark
    text = text.encode("ascii", "ignore").decode()      # drop the accent marks, emoji, etc.
    text = re.sub(r"[^a-z]", "", text.lower())          # keep a-z only
    return text[:MAX_SEED_LEN]


def is_letters(name):
    return LETTERS.fullmatch(name) is not None


def make_pair(seed):
    # Sample until the two names differ and are letters only. A vote between
    # two identical names teaches DPO nothing, and the model can in principle
    # sample the * or _ tokens mid-name.
    for _ in range(MAX_TRIES):
        name_1 = generate_name(model, seed)
        name_2 = generate_name(model, seed)
        if name_1 != name_2 and is_letters(name_1) and is_letters(name_2):
            return name_1, name_2
    return None


def new_pair(announce=True):
    # Called from the Generate button and after each vote.
    # The seed is saved here, at generation time, so editing the text box
    # before voting can't change which seed the vote is recorded against.
    typed = st.session_state.seed_box
    seed = clean_seed(typed)

    if announce and seed != typed.strip().lower():
        shown = seed.upper() if seed else "nothing (a fresh name)"
        st.session_state.message = ("info", f"Using letters only, up to {MAX_SEED_LEN}: starting from {shown}.")

    pair = make_pair(seed)
    if pair is None:
        st.session_state.pop("pair", None)
        st.session_state.message = ("info", "DinoGPT keeps giving the same name for this start. Try another one!")
        return

    st.session_state.seed = seed
    st.session_state.pair = pair


# ---------------------------------------------------------------
# Saving a vote: S3 when AWS secrets exist, else the local votes/ folder
# ---------------------------------------------------------------

def aws_config():
    # The [aws] section of .streamlit/secrets.toml, or None if there isn't one.
    # (Reading st.secrets with no secrets file at all raises an error.)
    try:
        return st.secrets["aws"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return None


@st.cache_resource
def s3_client():
    # One client per server, reused for every vote.
    aws = aws_config()
    return boto3.client(
        "s3",
        aws_access_key_id=aws["access_key_id"],
        aws_secret_access_key=aws["secret_access_key"],
        region_name=aws["region"],
    )


def is_valid_vote(vote):
    # Last check before anything leaves the app: only the shapes we expect.
    return (
        vote["preference"] in PREFERENCES
        and all(is_letters(vote[k]) for k in ("seed", "name_1", "name_2"))
        and len(vote["seed"]) <= MAX_SEED_LEN
        and all(len(vote[k]) <= 27 for k in ("name_1", "name_2"))
        and vote["name_1"] != vote["name_2"]
    )


def save_vote(vote):
    if not is_valid_vote(vote):
        raise ValueError(f"refusing to save malformed vote: {vote!r}")

    body = json.dumps(vote, indent=2)
    aws = aws_config()

    if aws is None:
        # No AWS secrets: save locally, for development only.
        VOTES_DIR.mkdir(exist_ok=True)
        (VOTES_DIR / f"{vote['vote_id']}.json").write_text(body)
        return

    # Same file name as before, under votes/ - the only place the
    # app's key is allowed to write.
    s3_client().put_object(
        Bucket=aws["bucket"],
        Key=f"votes/{vote['vote_id']}.json",
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )


def record_vote(preference):
    name_1, name_2 = st.session_state.pair
    vote = {
        "schema_version": 1,
        "vote_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": MODEL_FILE,
        "seed": st.session_state.seed,
        "name_1": name_1,
        "name_2": name_2,
        "preference": preference,
    }

    try:
        save_vote(vote)
    except Exception as e:
        # Keep the same pair on screen so the vote can simply be tried again.
        print(f"save_vote failed: {e!r}")   # shows in the Streamlit Cloud logs
        st.session_state.message = ("error", "Sorry, your vote couldn't be saved. Please try again.")
        return

    st.session_state.message = ("success", "Thanks! Here's another pair.")

    # Removing the pair first means a double click can't vote twice on it.
    del st.session_state.pair
    new_pair(announce=False)


# ---------------------------------------------------------------
# Page
# ---------------------------------------------------------------

st.title("DinoGPT Refinement")
st.markdown(
    "In [DinoGPT](https://dino-gpt.streamlit.app/) we built a small transformer "
    "that invents dinosaur names. Now we'd like your help to make it better: "
    "pick the name you like more, and we'll use the votes to fine-tune the model."
)

st.text_input(
    "Type the start of a dinosaur name (or leave it empty):",
    key="seed_box",
    max_chars=MAX_INPUT_LEN,
)
st.button("Generate Choices", icon="🦖", on_click=new_pair)

if "message" in st.session_state:
    kind, text = st.session_state.pop("message")
    getattr(st, kind)(text)

if "pair" in st.session_state:
    name_1, name_2 = st.session_state.pair

    st.subheader("Which name do you like better?")
    col_1, col_2 = st.columns(2)
    col_1.button(name_1.upper(), on_click=record_vote, args=("name_1",), width="stretch", type="primary")
    col_2.button(name_2.upper(), on_click=record_vote, args=("name_2",), width="stretch", type="primary")

    col_3, col_4 = st.columns(2)
    col_3.button("Both are fine", on_click=record_vote, args=("both_fine",), width="stretch")
    col_4.button("Hated both", on_click=record_vote, args=("both_bad",), width="stretch")

if aws_config() is None:
    st.caption("⚠️ No AWS secrets found: votes are being saved locally.")

with st.expander("What we collect, and what we don't"):
    st.markdown(
        """
        Each vote saves only this:

        - **The start you typed**, reduced to the letters a–z (`T-Rex` is saved as `trex`)
        - **The two names** DinoGPT suggested
        - **Your choice**: the first dino name, the second dino name, both are fine, or hated both
        - **The time** of the vote
        - **A random vote ID**, so votes don't overwrite each other
        - **Which version of the model** made the names

        That's all. This app doesn't ask for or record your name, email, IP address,
        location, or anything else about you, so votes are anonymous.

        The app is hosted on Streamlit Community Cloud, which may collect its own usage
        data under [Streamlit's privacy policy](https://streamlit.io/privacy-policy).
        """
    )

st.divider()
st.caption(
    "© 2026 Shishir Dahake · Apache-2.0 · Built on "
    "[DinoGPT](https://github.com/shishirdahake/dino_gpt)"
)