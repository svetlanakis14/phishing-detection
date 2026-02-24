import numpy as np

SYNONYM_MAP = {
    'click': 'press', 'here': 'now', 'free': 'gratis', 'verify': 'confirm',
    'account': 'profile', 'password': 'passkey', 'urgent': 'immediate',
    'login': 'sign in', 'bank': 'financial institution', 'update': 'renew',
    'winner': 'recipient', 'prize': 'reward', 'suspended': 'limited',
    'confirm': 'validate', 'payment': 'transaction', 'invoice': 'bill',
}


def character_swap_attack(text, swap_prob = 0.4):
    char_map = {'l': '1', 'o': '0', 'a': '@', 'e': '3', 'i': '!'}
    words  = text.split()
    result = []
    rng = np.random.default_rng(42)
    for word in words:
        if rng.random() < swap_prob and len(word) > 3:
            new_word = ''
            for ch in word:
                new_word += char_map.get(ch, ch) if rng.random() < 0.7 else ch
            result.append(new_word)
        else:
            result.append(word)
    return ' '.join(result)


def synonym_attack(text):
    words = text.split()
    result = []

    for w in words:
        w_clean = w.lower().strip(".,!?")
        if w_clean in SYNONYM_MAP:
            result.append(SYNONYM_MAP[w_clean])
        else:
            result.append(w)

    return ' '.join(result)


def whitespace_attack(text):
    trigger_words = list(SYNONYM_MAP.keys())
    words  = text.split()
    result = []
    rng = np.random.default_rng(0)
    for word in words:
        if word in trigger_words and rng.random() < 0.5:
            result.append(' '.join(list(word)))
        else:
            result.append(word)
    return ' '.join(result)


def apply_adversarial_attacks(texts):
    return {
        'char_swap':  [character_swap_attack(t) for t in texts],
        'synonym':    [synonym_attack(t)         for t in texts],
        'whitespace': [whitespace_attack(t)      for t in texts],
    }