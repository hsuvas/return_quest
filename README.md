# Return Quest

An interactive role-playing game where you play a real Amazon customer trying to return a product — and negotiate your way to a fair resolution with an AI-powered customer service agent.

Built to study how LLMs handle policy ambiguity in multi-turn conversations.

---

## What It Is

You pick a character, choose products to return, and have a live chat with an LLM agent that follows Amazon's return policy. The agent asks clarifying questions, consults policy clauses, uses tools (order lookup, eligibility checks, etc.), and eventually reaches a formal resolution. You can accept it — or push back.

The scenarios are deliberately designed around **policy ambiguities**: edge cases where the correct outcome genuinely depends on how you read the policy. There is no single right answer. The agent has to reason under uncertainty, and so do you.

---

## Workflow

```
Step 1 — Choose Your Avatar
         Pick a persona (name, location, job, personality style)
         ↓
Step 2 — Select Items to Return
         Search 900+ Amazon products, pick 1–3 to return, assign a reason per item
         ↓
Step 3 — Choose Your Agent & Model
         Five agent personalities (Direct → Very Helpful) + GPT model selector
         ↓
Step 4 — Your Mission
         An LLM generates a complex return scenario based on your products
         and randomly-selected policy ambiguities. Read it, then write your opening message.
         ↓
Step 5 — Live Conversation
         Multi-turn chat. The agent calls tools, reasons about policy, asks questions.
         Use the Hint button for a nudge if you're stuck.
         ↓
Step 6 — Resolution
         The agent issues a formal resolution (full refund / partial / denial / escalation / etc.)
         Accept it, or reject it and keep negotiating.
```

---

## Agent Personalities

| Persona | Style | Difficulty |
|---|---|---|
| **Direct** | Strictly policy-based, no sugar-coating | Hard |
| **Fair** | Balanced, procedural, treats all cases equally | Medium |
| **Agreeable** | Warm, empathetic, seeks win-win | Easy |
| **Helpful** | Proactive, goes the extra mile | Easy |
| **Very Helpful** | Customer-first, willing to bend small rules | Very Easy |

---

## Resolution Types

| Outcome | Meaning |
|---|---|
| Full Refund to Bank | Full refund to original payment method |
| Partial Refund | Reduced refund based on item condition |
| Store Credit | Gift card instead of cash refund |
| Replacement / Exchange | Item replaced or swapped |
| Escalated | Handed off to a human agent |
| Denied | Return request rejected under policy |

---

## Architecture

```
return_quest/
├── app.py                  ← Streamlit UI (entry point)
├── server.py               ← FastAPI backend (session management, agent turns)
├── showcase_backend.py     ← Scenario builder, agent runner, LLM task generation
├── policy_ambiguities_v3_final.csv  ← Amazon policy ambiguity database
├── agent/                  ← Agent pipeline (LLM provider, tools, prompt builder, etc.)
├── data/
│   ├── product_details/    ← 900+ Amazon product descriptions
│   └── persona_hub/        ← Preset customer personas
├── images/                 ← Character and agent avatars
├── output/                 ← Saved session files (per conversation)
├── requirements.txt
└── run.sh                  ← Local setup + run script
```

**How a turn works:**

1. Player types a message → Streamlit sends it to the FastAPI backend (`POST /api/session/{id}/turn`)
2. The backend appends the message to the server-side `ConversationState` and calls `run_agent_turn()`
3. The agent LLM reasons, calls tools (order lookup, eligibility check, policy query, etc.), and returns a response
4. The FastAPI server returns `{agent_message, tool_calls, finished, resolution}` to the frontend
5. Streamlit renders the agent message and any tool call indicators
6. If `finished=True`, the resolution screen is shown

Sessions are stored server-side (in-memory). The frontend only holds a `session_id`.

**Hint system:**

Clicking "Get a Hint" calls `GET /api/session/{id}/hint`, which runs a lightweight LLM prompt that produces a one-sentence coaching nudge (not the full message) — e.g. *"Try asking about the return window for third-party sellers."*

**Session data:**

When a player accepts a resolution, the full session is saved to `output/{first_name}_{session_id}.json` with:
- Player profile
- Task details (products, return reasons, policy tensions)
- Full conversation history
- Resolution reached
- Stats (turns, tool calls, hints used)

---

## Running Locally

**Requirements:** Python 3.10+, an OpenAI API key

```bash
# Clone the repo, then:
cd return_quest

# Create a .env file with your API key
echo 'OPENAI_API_KEY=sk-...' > .env

# Run (creates venv + installs deps automatically)
./run.sh
```

Or manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (the `.env` file is gitignored — your key stays local)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, main file path `return_quest/app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
5. Click **Deploy**

Streamlit Community Cloud handles environment creation and dependency installation from `requirements.txt` automatically.

---

## Research Context

This game is built on top of a dataset generation pipeline that:

1. Scrapes and analyses policy documents to extract ambiguous clauses (anaphoric ambiguity, coordination ambiguity, missing conditions)
2. Generates complex, multi-outcome customer support scenarios that deliberately exploit those ambiguities
3. Simulates multi-turn conversations between LLM agents and LLM customers
4. Evaluates agent performance with reference-based and LLM-as-judge scoring

The interactive game replaces the simulated customer with a human player, making the policy ambiguities tangible and observable in real time.

Primary policy: [Amazon Return Policy](https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7)
