"""
t9_task.py - T9 Predictive Text Task
Defines the T9 task: given a sequence of keypad digits, predict the intended word.
Includes dictionary, encoding, training loop, and evaluation metrics.
"""
import numpy as np

# T9 keypad mapping
KEYPAD = {
    'a': 2, 'b': 2, 'c': 2,
    'd': 3, 'e': 3, 'f': 3,
    'g': 4, 'h': 4, 'i': 4,
    'j': 5, 'k': 5, 'l': 5,
    'm': 6, 'n': 6, 'o': 6,
    'p': 7, 'q': 7, 'r': 7, 's': 7,
    't': 8, 'u': 8, 'v': 8,
    'w': 9, 'x': 9, 'y': 9, 'z': 9,
}

# Common English words for T9 (curated to have interesting ambiguities)
WORD_LIST = [
    # 2-letter
    "an", "am", "at", "be", "by", "do", "go", "he", "if", "in",
    "is", "it", "me", "my", "no", "of", "on", "or", "so", "to",
    "up", "us", "we",
    # 3-letter
    "act", "add", "age", "ago", "air", "all", "and", "any", "arm",
    "art", "ask", "ate", "bad", "bag", "bar", "bed", "big", "bit",
    "box", "boy", "bug", "bus", "but", "buy", "can", "cap", "car",
    "cat", "cup", "cut", "dad", "day", "did", "die", "dig", "dog",
    "dry", "ear", "eat", "egg", "end", "eye", "fan", "far", "fat",
    "few", "fit", "fix", "fly", "for", "fun", "gas", "get", "god",
    "got", "gun", "guy", "had", "has", "hat", "her", "him", "his",
    "hit", "hot", "how", "ice", "ill", "its", "job", "key", "kid",
    "law", "lay", "leg", "let", "lie", "lip", "lot", "low", "mad",
    "man", "map", "may", "met", "mix", "mom", "mud", "net", "new",
    "nor", "not", "now", "nut", "odd", "off", "oil", "old", "one",
    "our", "out", "own", "pay", "pen", "per", "pet", "pie", "pin",
    "pop", "pot", "put", "ran", "raw", "red", "rid", "row", "run",
    "sad", "sat", "saw", "say", "sea", "set", "she", "sir", "sit",
    "six", "sky", "son", "sun", "ten", "the", "tie", "tip", "toe",
    "too", "top", "try", "two", "use", "van", "war", "was", "way",
    "wet", "who", "why", "win", "won", "yes", "yet", "you",
    # 4-letter
    "able", "also", "area", "army", "away", "back", "ball", "band",
    "bank", "base", "bath", "bear", "beat", "been", "bell", "best",
    "bill", "bird", "blow", "blue", "boat", "body", "bomb", "bone",
    "book", "born", "both", "burn", "busy", "call", "calm", "came",
    "camp", "card", "care", "case", "cast", "cell", "chin", "city",
    "club", "coat", "code", "cold", "come", "cook", "cool", "copy",
    "core", "cost", "crew", "crop", "dark", "data", "date", "dead",
    "deal", "dear", "deep", "door", "down", "draw", "drop", "drug",
    "dust", "duty", "each", "earn", "east", "edge", "else", "even",
    "ever", "evil", "face", "fact", "fail", "fair", "fall", "farm",
    "fast", "fate", "fear", "feed", "feel", "fell", "file", "fill",
    "film", "find", "fine", "fire", "firm", "fish", "five", "flat",
    "flow", "food", "foot", "form", "four", "free", "from", "fuel",
    "full", "fund", "gain", "game", "gave", "gift", "girl", "give",
    "glad", "goal", "goes", "gold", "golf", "gone", "good", "grab",
    "gray", "grew", "grow", "gulf", "hair", "half", "hall", "hand",
    "hang", "hard", "harm", "hate", "have", "head", "hear", "heat",
    "help", "here", "hero", "hide", "high", "hill", "hold", "hole",
    "holy", "home", "hope", "host", "hour", "huge", "hung", "hurt",
    "idea", "iron", "item", "jack", "join", "jump", "jury", "just",
    "keen", "keep", "kept", "kick", "kill", "kind", "king", "knee",
    "knew", "know", "lack", "lady", "laid", "lake", "land", "lane",
    "last", "late", "lead", "left", "less", "life", "lift", "like",
    "line", "link", "list", "live", "lock", "long", "look", "lord",
    "lose", "loss", "lost", "love", "luck", "made", "mail", "main",
    "make", "male", "mark", "mass", "meal", "mean", "meet", "mile",
    "mind", "mine", "miss", "mode", "mood", "moon", "more", "most",
    "move", "much", "must", "name", "near", "neck", "need", "news",
    "next", "nice", "nine", "none", "nose", "note", "odds", "once",
    "only", "onto", "open", "over", "pace", "pack", "page", "paid",
    "pain", "pair", "pale", "palm", "park", "part", "pass", "past",
    "path", "peak", "pick", "pile", "pine", "pink", "plan", "play",
    "plot", "plus", "poem", "poet", "poll", "pool", "poor", "port",
    "post", "pour", "pray", "pull", "pure", "push", "race", "rain",
    "rank", "rare", "rate", "read", "real", "rear", "rely", "rest",
    "rice", "rich", "ride", "ring", "rise", "risk", "road", "rock",
    "role", "roll", "roof", "room", "root", "rope", "rose", "rule",
    "rush", "safe", "said", "sake", "sale", "salt", "same", "sand",
    "sang", "save", "seat", "seed", "seek", "seem", "seen", "self",
    "sell", "send", "sent", "ship", "shop", "shot", "show", "shut",
    "sick", "side", "sign", "sing", "sink", "site", "size", "skin",
    "slip", "slow", "snow", "soft", "soil", "sold", "sole", "some",
    "song", "soon", "sort", "soul", "spin", "spot", "star", "stay",
    "step", "stop", "such", "suit", "sure", "swim", "tail", "take",
    "tale", "talk", "tall", "tank", "tape", "task", "team", "tell",
    "tend", "term", "test", "text", "than", "that", "them", "then",
    "they", "thin", "this", "thus", "till", "time", "tiny", "tire",
    "told", "tone", "took", "tool", "tops", "torn", "tour", "town",
    "tree", "trip", "true", "turn", "twin", "type", "unit", "upon",
    "used", "user", "vast", "very", "vice", "view", "vote", "wage",
    "wait", "wake", "walk", "wall", "want", "warm", "warn", "wash",
    "wave", "weak", "wear", "week", "well", "went", "were", "west",
    "what", "when", "whom", "wide", "wife", "wild", "will", "wind",
    "wine", "wing", "wire", "wise", "wish", "with", "wood", "word",
    "wore", "work", "worm", "worn", "wrap", "yard", "yeah", "year",
    "your", "zone",
]

# Word frequencies (approximate, log-scaled rank, higher = more common)
# Top words get highest scores
FREQ_RANKS = {}
_common = ["the", "and", "that", "have", "for", "not", "with", "you",
           "this", "but", "his", "from", "they", "been", "her", "she",
           "one", "all", "just", "like", "come", "make", "time", "back",
           "good", "know", "take", "year", "them", "some", "than", "over",
           "very", "when", "also", "into", "your", "work", "will", "well",
           "more", "what", "most", "only", "call", "find", "here", "each"]
for i, w in enumerate(_common):
    FREQ_RANKS[w] = 1.0 - i * 0.015


def word_to_digits(word):
    """Convert word to T9 digit sequence."""
    digits = []
    for ch in word.lower():
        if ch in KEYPAD:
            digits.append(KEYPAD[ch])
    return tuple(digits)


def build_t9_dataset():
    """
    Build T9 dataset: digit sequences -> word indices.
    Returns: digit_to_words (dict), word_to_idx (dict), idx_to_word (dict),
             all_words (list), bigrams (dict)
    """
    word_to_idx = {}
    idx_to_word = {}
    digit_to_words = {}

    for i, w in enumerate(WORD_LIST):
        word_to_idx[w] = i
        idx_to_word[i] = w
        digits = word_to_digits(w)
        if digits not in digit_to_words:
            digit_to_words[digits] = []
        digit_to_words[digits].append(i)

    # Simple bigram model (synthetic but structured)
    bigrams = {}
    # Common bigram patterns
    pairs = [
        ("the", "end"), ("the", "way"), ("the", "man"), ("the", "old"),
        ("in", "the"), ("of", "the"), ("to", "the"), ("on", "the"),
        ("is", "not"), ("do", "not"), ("can", "not"), ("it", "is"),
        ("he", "was"), ("she", "was"), ("we", "are"), ("you", "are"),
        ("go", "back"), ("come", "back"), ("get", "out"), ("come", "in"),
        ("at", "home"), ("in", "time"), ("by", "the"), ("for", "the"),
    ]
    for w1, w2 in pairs:
        if w1 in word_to_idx and w2 in word_to_idx:
            bigrams[(word_to_idx[w1], word_to_idx[w2])] = 2.0

    return digit_to_words, word_to_idx, idx_to_word, WORD_LIST, bigrams


class T9Task:
    """T9 prediction task for evaluating nucleated networks."""

    MAX_DIGITS = 6  # max word length we handle
    N_DIGITS = 8    # digits 2-9

    def __init__(self):
        (self.digit_to_words, self.word_to_idx, self.idx_to_word,
         self.words, self.bigrams) = build_t9_dataset()
        self.vocab_size = len(self.words)

        # Build training data
        self._build_data()

    @property
    def input_size(self):
        return self.N_DIGITS * self.MAX_DIGITS  # one-hot digits, padded

    @property
    def output_size(self):
        return self.vocab_size

    def _build_data(self):
        """Build training pairs: encoded digits -> word index."""
        self.X = []
        self.y = []
        self.sample_weights = []

        for digits, word_indices in self.digit_to_words.items():
            x = self.encode_digits(digits)
            for wi in word_indices:
                self.X.append(x)
                self.y.append(wi)
                # Weight by frequency
                w = self.words[wi]
                freq = FREQ_RANKS.get(w, 0.3)
                self.sample_weights.append(freq)

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.sample_weights = np.array(self.sample_weights)
        self.sample_weights /= self.sample_weights.sum()

    def encode_digits(self, digits):
        """One-hot encode a digit sequence, padded to MAX_DIGITS."""
        x = np.zeros(self.N_DIGITS * self.MAX_DIGITS)
        for i, d in enumerate(digits[:self.MAX_DIGITS]):
            idx = d - 2  # digits 2-9 -> indices 0-7
            if 0 <= idx < self.N_DIGITS:
                x[i * self.N_DIGITS + idx] = 1.0
        return x

    def train_epoch(self, network, lr=0.01, batch_size=64):
        """One epoch of SGD training. Returns average loss."""
        n = len(self.X)
        indices = np.random.permutation(n)
        total_loss = 0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = self.X[batch_idx]
            y_batch = self.y[batch_idx]
            bs = len(batch_idx)

            # Forward
            probs = network.forward(X_batch)

            # Cross-entropy loss
            probs_clipped = np.clip(probs, 1e-10, 1.0)
            loss = -np.mean(np.log(probs_clipped[np.arange(bs), y_batch]))
            total_loss += loss

            # Backward (numerical gradients for simplicity + correctness)
            params = network.get_params()
            grad = np.zeros_like(params)

            # Analytical gradient for output layer at least
            # dL/dz = probs - one_hot(y)
            # For simplicity, use param perturbation for small networks
            eps = 1e-4
            for p_idx in range(len(params)):
                params_plus = params.copy()
                params_plus[p_idx] += eps
                network.set_params(params_plus)
                probs_plus = network.forward(X_batch)
                loss_plus = -np.mean(
                    np.log(np.clip(probs_plus[np.arange(bs), y_batch], 1e-10, 1)))

                grad[p_idx] = (loss_plus - loss) / eps

            # Update
            params -= lr * grad
            network.set_params(params)

            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_epoch_fast(self, network, lr=0.01, batch_size=64):
        """
        Faster training using analytical gradients for the output layer
        and finite differences for hidden layers.
        """
        n = len(self.X)
        indices = np.random.permutation(n)
        total_loss = 0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_b = self.X[batch_idx]
            y_b = self.y[batch_idx]
            bs = len(batch_idx)

            # Forward pass, capturing activations
            h = X_b
            activations = [h]
            pre_activations = []

            for i in range(network.n_layers):
                z = h @ network.weights[i] + network.biases[i]
                z = np.clip(z, -50, 50)  # prevent overflow
                pre_activations.append(z)

                if i < network.n_layers - 1:
                    h = np.maximum(0, z)
                else:
                    z_s = z - np.max(z, axis=-1, keepdims=True)
                    exp_z = np.exp(z_s)
                    h = exp_z / (np.sum(exp_z, axis=-1, keepdims=True) + 1e-10)
                activations.append(h)

            probs = activations[-1]
            probs_clipped = np.clip(probs, 1e-10, 1.0)
            loss = -np.mean(np.log(probs_clipped[np.arange(bs), y_b]))
            total_loss += loss

            # Backprop
            # Output gradient: dL/dz = (probs - one_hot) / bs
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(bs), y_b] = 1.0
            delta = (probs - one_hot) / bs

            for i in range(network.n_layers - 1, -1, -1):
                # Gradient for weights[i] and biases[i]
                dW = activations[i].T @ delta
                db = delta.sum(axis=0)

                # Gradient clipping to prevent overflow in deep networks
                gn = np.linalg.norm(dW)
                if gn > 5.0:
                    dW = dW * (5.0 / gn)
                    db = db * (5.0 / gn)

                if np.any(np.isnan(dW)):
                    break  # bail on NaN gradients

                network.weights[i] -= lr * dW
                network.biases[i] -= lr * db

                if i > 0:
                    delta = delta @ network.weights[i].T
                    # ReLU derivative
                    delta *= (pre_activations[i - 1] > 0).astype(float)

            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, network):
        """
        Evaluate T9 performance.
        Returns: accuracy@1, accuracy@3, mean reciprocal rank
        """
        probs = network.forward(self.X)
        predictions = np.argsort(-probs, axis=1)

        acc1 = 0
        acc3 = 0
        mrr = 0

        for i in range(len(self.y)):
            true = self.y[i]
            preds = predictions[i]
            if preds[0] == true:
                acc1 += 1
            if true in preds[:3]:
                acc3 += 1
            rank = np.where(preds == true)[0]
            if len(rank) > 0:
                mrr += 1.0 / (rank[0] + 1)

        n = len(self.y)
        return {
            'acc1': acc1 / n,
            'acc3': acc3 / n,
            'mrr': mrr / n,
        }

    def evaluate_ambiguity(self, network):
        """
        Specifically test T9 ambiguity resolution.
        Only looks at digit sequences that map to multiple words.
        """
        ambiguous = {d: ws for d, ws in self.digit_to_words.items()
                     if len(ws) > 1}

        correct = 0
        total = 0
        for digits, word_indices in ambiguous.items():
            x = self.encode_digits(digits)
            probs = network.forward(x.reshape(1, -1))[0]

            # The "correct" word is the most frequent one
            best_freq = -1
            best_idx = word_indices[0]
            for wi in word_indices:
                f = FREQ_RANKS.get(self.words[wi], 0.3)
                if f > best_freq:
                    best_freq = f
                    best_idx = wi

            if np.argmax(probs) == best_idx:
                correct += 1
            total += 1

        return correct / max(total, 1)
