# DinoGPT Refinement
 
*Episode 2 of [DinoGPT](https://github.com/shishirdahake/dino_gpt): teaching a tiny transformer which dinosaur names people actually like.*
 
> 🚧 **Work in progress.** This README will grow as the project does.
 
In episode 1, I built DinoGPT from scratch: a character-level transformer (1 block, 4 heads, 53,533 parameters) trained on 1,535 dinosaur names. [Try it live](https://dino-gpt.streamlit.app/).
 
Episode 2 fine-tunes it with **human feedback**. A voting app shows two names generated from the same starting letters, people pick the one they prefer, and those votes are used to fine-tune the model with **DPO** (Direct Preference Optimization).
 
## Plan
 
1. ✅ **Collect votes.** A Streamlit app shows two names per seed and saves each vote to S3.
2. ⬜ **Fine-tune with DPO.** Start from `dino_gpt_original.pth`, keep a frozen copy as the reference model, and save the result as `dino_gpt_dpo.pth`.
3. ⬜ **Compare.** Show the original and fine-tuned models side by side.
## Files
 
| File | What it is |
|---|---|
| `voting_app.py` | Streamlit voting app |
| `dinohelper.py` | The model classes and `generate_name`, from episode 1 |
| `encoder.py` | Character tokenizer: a–z plus `*` (start), `$` (end), `_` (padding) |
| `dino_gpt_original.pth` | Episode 1 weights, the starting point for fine-tuning |
| `requirements.txt` | Packages the app needs |
 
## Running the voting app
 
```bash
pip install -r requirements.txt
streamlit run voting_app.py
```
 
Votes are saved to S3 when `.streamlit/secrets.toml` has an `[aws]` section:
 
```toml
[aws]
access_key_id = "..."
secret_access_key = "..."
region = "ap-south-1"
bucket = "your-bucket-name"
```
 
Without it, votes are saved to a local `votes/` folder and the page shows a warning. `secrets.toml` is git-ignored and must never be committed.
 
The app's AWS key can only **write** under `votes/` in one bucket. It can't list, read or delete anything. Votes are read back for training with a separate read-only key.
 
## What a vote looks like
 
One JSON file per vote, named by its vote ID:
 
```json
{
  "schema_version": 1,
  "vote_id": "3f2b8c1e-6d4a-4f7e-9a51-2c8e0b7d4a19",
  "timestamp": "2026-10-04T12:24:22Z",
  "model": "dino_gpt_original.pth",
  "seed": "silver",
  "name_1": "silverovenator",
  "name_2": "silverolesaurus",
  "preference": "name_1"
}
```
 
`preference` is one of `name_1`, `name_2`, `both_fine` or `both_bad`. Only `name_1` and `name_2` votes become DPO pairs. The ties are kept for analysis.
 
**Privacy:** a vote holds only the fields above. The app doesn't record names, emails, IP addresses or anything else about the person voting.
 
## License
 
Apache-2.0. Builds on [DinoGPT](https://github.com/shishirdahake/dino_gpt) by Shishir Dahake.