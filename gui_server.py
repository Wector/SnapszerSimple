import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import argparse
import json
from typing import List, Optional, Dict

# Import game logic
from snapszer_base import SnapszerState, TOTAL_ACTIONS, OBSERVATION_SIZE, card_str, card_id, SUIT_NAMES, RANK_NAMES, EXCHANGE_TRUMP_ACTION, CLOSE_TALON_ACTION, NUM_CARDS

app = FastAPI()

# --- Agent Architecture (Must match training) ---
class Agent(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, TOTAL_ACTIONS),
        )

    def get_action_and_value(self, x, action=None, legal_actions_mask=None):
        logits = self.actor(x)
        if legal_actions_mask is not None:
            logits = logits.masked_fill(legal_actions_mask == 0, -1e9)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

# --- Global State ---
class GameContext:
    def __init__(self):
        self.agent: Optional[Agent] = None
        self.device: str = "cpu"
        self.state: Optional[SnapszerState] = None
        self.human_player_id: int = 0
        self.metrics_path: str = "models/snapszer_ppo/metrics.jsonl"
        self.last_trick_cards: Optional[List[int]] = None
        self.last_trick_leader: Optional[int] = None
        self.last_marriage: Optional[Dict] = None # {player: int, suit: int, points: int}
        self.model_path: Optional[str] = None
        self.model_mtime: float = 0.0

ctx = GameContext()

# --- Models ---
class ActionRequest(BaseModel):
    action: int

class CardData(BaseModel):
    id: int
    suit: int
    rank: int
    name: str
    image: str # Placeholder for now, or css class

class GameStateResponse(BaseModel):
    human_hand: List[CardData]
    opponent_hand_count: int
    trick_cards: List[Optional[CardData]]
    trump_card: Optional[CardData]
    trump_suit: int
    stock_count: int
    points: List[int] # [human, agent]
    game_points: Optional[List[int]]
    tricks_won: List[int]
    closed: bool
    trump_taken: bool
    leader: int
    current_player: int
    legal_actions: List[int]
    last_trick_winner: Optional[int]
    last_trick_cards: Optional[List[CardData]]
    last_trick_leader: Optional[int]
    last_marriage: Optional[Dict]
    terminal: bool
    winner: Optional[int]
    message: str

# --- Helpers ---
def get_card_data(cid: Optional[int]) -> Optional[CardData]:
    if cid is None: return None
    # suit: 0=S, 1=H, 2=D, 3=C
    # rank: 0=A, 1=10, 2=K, 3=Q, 4=J
    s = cid // 5
    r = cid % 5
    return CardData(
        id=cid,
        suit=s,
        rank=r,
        name=card_str(cid),
        image=f"card-{s}-{r}" # CSS class
    )

def build_response(state: SnapszerState, msg: str = "") -> GameStateResponse:
    human = ctx.human_player_id
    agent = 1 - human
    
    legal = state.legal_actions() if state.current_player == human else []
    
    last_trick = None
    if ctx.last_trick_cards:
        last_trick = [get_card_data(c) for c in ctx.last_trick_cards]

    return GameStateResponse(
        human_hand=[get_card_data(c) for c in sorted(state.hands[human])],
        opponent_hand_count=len(state.hands[agent]),
        trick_cards=[get_card_data(state.trick_cards[0]), get_card_data(state.trick_cards[1])],
        trump_card=get_card_data(state.trump_card) if not state.trump_taken else None,
        trump_suit=state.trump,
        stock_count=state.stock_remaining(),
        points=[state.points[human], state.points[agent]],
        game_points=list(state.game_points) if state.game_points else None,
        tricks_won=[state.tricks_won[human], state.tricks_won[agent]],
        closed=state.closed,
        trump_taken=state.trump_taken,
        leader=state.leader,
        current_player=state.current_player,
        legal_actions=legal,
        last_trick_winner=state.last_trick_winner,
        last_trick_cards=last_trick,
        last_trick_leader=ctx.last_trick_leader,
        last_marriage=ctx.last_marriage,
        terminal=state.terminal,
        winner=state.winner if state.winner is not None else None,
        message=msg
    )

def apply_action_with_tracking(action: int):
    # Reset transient events
    ctx.last_marriage = None
    
    # Check if this action completes a trick
    if action < 20 and ctx.state.trick_cards[0] is not None:
        ctx.last_trick_cards = [ctx.state.trick_cards[0], action]
        ctx.last_trick_leader = ctx.state.leader
    else:
        if ctx.state.trick_cards[0] is None:
             ctx.last_trick_cards = None
             ctx.last_trick_leader = None
    
    # Capture marriage state before action
    import copy
    prev_marriages = copy.deepcopy(ctx.state.marriages_scored)
    player = ctx.state.current_player
    
    ctx.state.apply_action(action)
    
    # Check for new marriages
    for s in range(4):
        if ctx.state.marriages_scored[player][s] and not prev_marriages[player][s]:
            points = 40 if s == ctx.state.trump else 20
            ctx.last_marriage = {"player": player, "suit": s, "points": points}
            break

# --- API Endpoints ---

@app.post("/api/new_game")
async def new_game():
    # Check for model update
    if ctx.model_path and os.path.exists(ctx.model_path):
        try:
            mtime = os.path.getmtime(ctx.model_path)
            if mtime > ctx.model_mtime:
                print(f"Model file changed. Reloading from {ctx.model_path}...")
                state_dict = torch.load(ctx.model_path, map_location=ctx.device)
                ctx.agent.load_state_dict(state_dict)
                ctx.model_mtime = mtime
                print("Model reloaded successfully.")
        except Exception as e:
            print(f"Failed to reload model: {e}")

    ctx.state = SnapszerState.new()
    ctx.last_trick_cards = None
    ctx.last_trick_leader = None
    ctx.human_player_id = 0 # Human is always player 0 for now
    # If agent starts, we do NOT move automatically anymore. Frontend must request it.
    msg = "New game started."
    if ctx.state.current_player != ctx.human_player_id:
        msg = "New game started. Agent's turn."
    return build_response(ctx.state, msg)

@app.get("/api/state")
async def get_state():
    if ctx.state is None:
        ctx.state = SnapszerState.new()
    return build_response(ctx.state)

@app.post("/api/action")
async def take_action(req: ActionRequest):
    if ctx.state is None:
        raise HTTPException(status_code=400, detail="Game not started")
    
    if ctx.state.current_player != ctx.human_player_id:
        raise HTTPException(status_code=400, detail="Not your turn")
    
    if req.action not in ctx.state.legal_actions():
        raise HTTPException(status_code=400, detail="Illegal action")
    
    try:
        apply_action_with_tracking(req.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Do NOT run agent turn automatically
    return build_response(ctx.state, "Action applied")

@app.post("/api/agent_step")
async def agent_step():
    if ctx.state is None or ctx.state.terminal:
        return build_response(ctx.state if ctx.state else SnapszerState.new(), "Game over")
        
    if ctx.state.current_player == ctx.human_player_id:
        return build_response(ctx.state, "It is your turn")

    # Agent turn
    obs = torch.tensor(ctx.state.observation_tensor(ctx.state.current_player), dtype=torch.float32).to(ctx.device).unsqueeze(0)
    legal = ctx.state.legal_actions()
    mask = torch.zeros((1, TOTAL_ACTIONS)).to(ctx.device)
    for act in legal:
        mask[0, act] = 1.0
    
    with torch.no_grad():
        action_tensor, _, _, _ = ctx.agent.get_action_and_value(obs, legal_actions_mask=mask)
        action = action_tensor.item()
    
    apply_action_with_tracking(action)
    
    return build_response(ctx.state, f"Agent played {action}")

def find_latest_metrics_file():
    models_dir = "models"
    if not os.path.exists(models_dir):
        return None
        
    latest_file = None
    latest_time = 0
    
    for root, dirs, files in os.walk(models_dir):
        if "metrics.jsonl" in files:
            path = os.path.join(root, "metrics.jsonl")
            try:
                mtime = os.path.getmtime(path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_file = path
            except OSError:
                pass
                
    return latest_file

@app.get("/api/metrics")
async def get_metrics():
    # Auto-detect latest metrics file if we are in "auto" mode (which we are by default now)
    # We can still respect ctx.metrics_path if it was explicitly set to something specific,
    # but for the user's request "don't have to fix it every time", auto-detection is best.
    
    # Let's prioritize the auto-detected latest file.
    metrics_path = find_latest_metrics_file()
    
    # Fallback to context path if nothing found (e.g. no experiments yet)
    if not metrics_path:
        metrics_path = ctx.metrics_path
        
    if not metrics_path or not os.path.exists(metrics_path):
        return JSONResponse(content=[])
    
    data = []
    try:
        with open(metrics_path, "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"Error reading metrics from {metrics_path}: {e}")
        return JSONResponse(content=[])
        
    # Add header to tell frontend which file we are serving (for debugging/display)
    headers = {"X-Metrics-Source": metrics_path, "Cache-Control": "no-cache, no-store, must-revalidate"}
    return JSONResponse(content=data, headers=headers)

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

# --- Static Files ---
app.mount("/", StaticFiles(directory="static", html=True), name="static")

def start_server(model_path: str, hidden_size: int, device: str, port: int):
    print(f"Loading model from {model_path}...")
    ctx.device = device
    ctx.model_path = model_path
    
    # Determine metrics path from model path
    # If model_path is "models/exp_name/latest.pt", metrics should be "models/exp_name/metrics.jsonl"
    model_dir = os.path.dirname(model_path)
    ctx.metrics_path = os.path.join(model_dir, "metrics.jsonl")
    print(f"Monitoring metrics from: {ctx.metrics_path}")
    
    ctx.agent = Agent(hidden_size=hidden_size).to(device)
    # Load model - handle potential map_location issues
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            ctx.agent.load_state_dict(state_dict)
            ctx.model_mtime = os.path.getmtime(model_path)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"WARNING: Failed to load model from {model_path}: {e}")
            print("Starting with a random initialized agent.")
    else:
        print(f"WARNING: Model file {model_path} not found. Starting with random agent.")
        
    ctx.agent.eval()
    
    ctx.state = SnapszerState.new()
    
    print(f"Starting server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/snapszer_ppo/latest.pt", help="Path to model .pt file")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden size of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    start_server(args.model, args.hidden_size, args.device, args.port)
