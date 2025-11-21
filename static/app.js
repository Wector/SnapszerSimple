const API_URL = '/api';

// Game State
let gameState = null;
let isProcessing = false;

// DOM Elements
const els = {
    humanScore: document.getElementById('human-score'),
    agentScore: document.getElementById('agent-score'),
    humanGp: document.getElementById('human-gp'),
    agentGp: document.getElementById('agent-gp'),
    statusMessage: document.getElementById('status-message'),
    opponentHand: document.getElementById('opponent-hand'),
    opponentTricks: document.getElementById('opponent-tricks'),
    stockCount: document.getElementById('stock-count'),
    trumpCardContainer: document.getElementById('trump-card-container'),
    trumpSuitDisplay: document.getElementById('trump-suit-display'),
    trickCard0: document.getElementById('trick-card-0'),
    trickCard1: document.getElementById('trick-card-1'),
    playerHand: document.getElementById('player-hand'),
    playerTricks: document.getElementById('player-tricks'),
    btnExchange: document.getElementById('btn-exchange'),
    btnClose: document.getElementById('btn-close'),
    btnNewGame: document.getElementById('btn-new-game'),
    modalOverlay: document.getElementById('modal-overlay'),
    modalTitle: document.getElementById('modal-title'),
    modalMessage: document.getElementById('modal-message'),
    btnRestart: document.getElementById('btn-restart'),
};

// Constants
const SUIT_SYMBOLS = ['🌸', '☀️', '🍇', '❄️']; // Spring, Summer, Autumn, Winter
const SUIT_NAMES = ['Tavasz', 'Nyár', 'Ősz', 'Tél'];
const SUIT_CLASSES = ['suit-spring', 'suit-summer', 'suit-autumn', 'suit-winter'];
const RANK_NAMES = ['Ász', 'Tíz', 'Király', 'Felső', 'Alsó'];

// Init
async function init() {
    els.btnNewGame.addEventListener('click', startNewGame);
    els.btnRestart.addEventListener('click', startNewGame);
    els.btnExchange.addEventListener('click', () => sendAction(20)); // 20 = Exchange
    els.btnClose.addEventListener('click', () => sendAction(21)); // 21 = Close

    await fetchState();
}

async function fetchState() {
    try {
        const res = await fetch(`${API_URL}/state`);
        const data = await res.json();
        handleStateUpdate(data);
    } catch (e) {
        console.error("Failed to fetch state", e);
        els.statusMessage.textContent = "Error connecting to server.";
    }
}

async function startNewGame() {
    isProcessing = false;
    try {
        els.modalOverlay.classList.add('hidden');
        const res = await fetch(`${API_URL}/new_game`, { method: 'POST' });
        const data = await res.json();
        handleStateUpdate(data);
    } catch (e) {
        console.error("Failed to start new game", e);
        isProcessing = false;
    }
}

async function sendAction(actionId) {
    if (isProcessing) return;
    isProcessing = true;
    render(gameState); // Re-render to disable cards
    try {
        const res = await fetch(`${API_URL}/action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: actionId })
        });

        if (!res.ok) {
            const err = await res.json();
            alert(err.detail);
            isProcessing = false;
            render(gameState);
            return;
        }

        const data = await res.json();
        await handleStateUpdate(data);
    } catch (e) {
        console.error("Failed to send action", e);
        isProcessing = false;
        render(gameState);
    }
}

async function triggerAgentStep() {
    try {
        els.statusMessage.textContent = "Agent is thinking...";
        // Small delay before request to simulate thinking
        await new Promise(r => setTimeout(r, 500));

        const res = await fetch(`${API_URL}/agent_step`, { method: 'POST' });
        if (!res.ok) {
            throw new Error(`Agent step failed: ${res.statusText}`);
        }
        const data = await res.json();
        await handleStateUpdate(data);
    } catch (e) {
        console.error("Failed to trigger agent step", e);
        els.statusMessage.textContent = "Error: Agent failed to move.";
        isProcessing = false;
        render(gameState);
    }
}

async function handleStateUpdate(state) {
    // If there was a last trick, show it first
    if (state.last_trick_cards && state.last_trick_cards.length > 0) {
        // Render the state first to update scores/hands, but then override the trick area
        render(state);
        renderTrick(state.last_trick_cards, state.leader);

        // Wait for user to see the trick (2 seconds)
        await new Promise(r => setTimeout(r, 2000));
    }

    render(state);

    if (state.terminal) {
        isProcessing = false;
        return;
    }

    if (state.current_player === 1) { // Agent's turn
        await triggerAgentStep();
    } else {
        isProcessing = false;
        render(state); // Re-enable controls
    }
}

function createCardEl(cardData, onClick = null, disabled = false, marriagePoints = null) {
    const el = document.createElement('div');
    el.className = `card ${disabled ? 'disabled' : ''}`;

    if (!cardData) {
        return el;
    }

    const suitSym = SUIT_SYMBOLS[cardData.suit];
    const suitClass = SUIT_CLASSES[cardData.suit];
    const rankName = RANK_NAMES[cardData.rank];

    // Set content first
    el.innerHTML = `
        <div class="card-content ${suitClass}">
            <div class="card-top">
                <span class="rank">${rankName}</span>
            </div>
            <div class="card-center">
                <span class="suit-icon-large">${suitSym}</span>
            </div>
        </div>
    `;

    // Then append badge if needed
    if (marriagePoints) {
        el.classList.add('marriage-highlight');
        const badge = document.createElement('div');
        badge.className = 'marriage-badge';
        badge.textContent = `+${marriagePoints}`;
        el.appendChild(badge);
    }

    if (onClick && !disabled) {
        el.addEventListener('click', onClick);
    }

    return el;
}

function renderTrick(cards, leader) {
    els.trickCard0.innerHTML = '';
    els.trickCard1.innerHTML = '';

    if (cards[0]) els.trickCard0.appendChild(createCardEl(cards[0]));
    if (cards[1]) els.trickCard1.appendChild(createCardEl(cards[1]));
}

function render(state) {
    gameState = state;

    // Scores
    els.humanScore.textContent = state.points[0];
    els.agentScore.textContent = state.points[1];
    els.humanGp.textContent = state.game_points ? `GP: ${state.game_points[0]}` : "GP: 0";
    els.agentGp.textContent = state.game_points ? `GP: ${state.game_points[1]}` : "GP: 0";
    els.statusMessage.textContent = state.message || getStatusText(state);

    // Tricks
    els.playerTricks.textContent = state.tricks_won[0];
    els.opponentTricks.textContent = state.tricks_won[1];

    // Stock
    els.stockCount.textContent = state.stock_count;

    // Trump
    els.trumpCardContainer.innerHTML = '';
    if (state.trump_card) {
        const trumpEl = createCardEl(state.trump_card);
        els.trumpCardContainer.appendChild(trumpEl);
        els.trumpSuitDisplay.innerHTML = '';
    } else {
        const suitSym = SUIT_SYMBOLS[state.trump_suit];
        const suitClass = SUIT_CLASSES[state.trump_suit];
        els.trumpSuitDisplay.innerHTML = `<span class="${suitClass}" style="font-size: 2rem; opacity: 0.5">${suitSym}</span>`;
    }

    // Opponent Hand
    els.opponentHand.innerHTML = '';
    for (let i = 0; i < state.opponent_hand_count; i++) {
        const card = document.createElement('div');
        card.className = 'card back';
        els.opponentHand.appendChild(card);
    }

    // Identify Marriages for Highlighting
    const marriageCards = new Set(); // Set of card IDs
    const marriagePointsMap = {}; // Map card ID -> points

    const suits = [[], [], [], []];
    state.human_hand.forEach(c => suits[c.suit].push(c));

    suits.forEach((cards, suitIdx) => {
        const ranks = cards.map(c => c.rank);
        // King is 2, Queen (Felső) is 3
        if (ranks.includes(2) && ranks.includes(3)) {
            const points = (suitIdx === state.trump_suit) ? 40 : 20;
            cards.forEach(c => {
                if (c.rank === 2 || c.rank === 3) {
                    marriageCards.add(c.id);
                    marriagePointsMap[c.id] = points;
                }
            });
        }
    });

    // Player Hand
    els.playerHand.innerHTML = '';
    state.human_hand.forEach(card => {
        const isLegal = state.legal_actions.includes(card.id);
        const disabled = !isLegal || isProcessing || state.current_player !== 0;

        let points = null;
        if (marriageCards.has(card.id)) {
            points = marriagePointsMap[card.id];
        }

        const el = createCardEl(card, () => sendAction(card.id), disabled, points);
        els.playerHand.appendChild(el);
    });

    // Trick Area
    renderTrick(state.trick_cards, state.leader);

    // Actions
    const myTurn = state.current_player === 0 && !isProcessing;
    els.btnExchange.disabled = !myTurn || !state.legal_actions.includes(20);
    els.btnClose.disabled = !myTurn || !state.legal_actions.includes(21);

    // Game Over
    if (state.terminal && !state.last_trick_cards) {
        showGameOver(state);
    }
}

function showGameOver(state) {
    els.modalOverlay.classList.remove('hidden');
    if (state.winner === 0) {
        els.modalTitle.textContent = "You Won!";
        els.modalMessage.textContent = `Score: ${state.points[0]} - ${state.points[1]}`;
    } else {
        els.modalTitle.textContent = "Agent Won";
        els.modalMessage.textContent = `Score: ${state.points[0]} - ${state.points[1]}`;
    }
}

function getStatusText(state) {
    if (state.terminal) return "Game Over";
    if (state.current_player === 0) return "Your Turn";
    return "Agent's Turn";
}

function showNotification(message) {
    const notif = document.createElement('div');
    notif.className = 'notification';
    notif.textContent = message;
    document.body.appendChild(notif);

    // Trigger animation
    setTimeout(() => notif.classList.add('show'), 10);

    // Remove after 3 seconds
    setTimeout(() => {
        notif.classList.remove('show');
        setTimeout(() => notif.remove(), 300);
    }, 3000);
}

init();
