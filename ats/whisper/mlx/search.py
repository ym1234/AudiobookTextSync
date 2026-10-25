"""the decode loop for the mlx whisper, on the gpu from the first token to the last"""
import numpy as np
import mlx.core as mx

_SYNC_EVERY = 4 # how often the sampling search checks whether every sequence has ended; checking every step costs
                # ~25% (it blocks until the gpu drains), checking rarely wastes decode steps past the end

class LogitRules:
    """whisper's rules for the next token, as masks over the whole batch so they never leave the gpu"""

    def __init__(self, tokenizer, n_vocab, suppress, blank, max_initial_timestamp_index, repetition_penalty):
        self.ts_begin, self.eot = tokenizer.timestamp_begin, tokenizer.eot
        self.repetition_penalty = repetition_penalty
        self.idx = mx.arange(n_vocab)
        self.is_ts, self.is_text, self.is_before_eot = self.idx >= self.ts_begin, self.idx < self.ts_begin, self.idx < self.eot

        base = np.zeros(n_vocab, dtype=np.float32)
        base[suppress] = -np.inf
        first = base.copy()
        if blank is not None: first[blank] = -np.inf
        first[:self.ts_begin] = -np.inf # the first token is always a timestamp
        if max_initial_timestamp_index is not None:
            first[self.ts_begin+int(max_initial_timestamp_index)+1:] = -np.inf
        self.base, self.first = mx.array(base), mx.array(first)

    def first_step(self, lg):
        """lg (rows, vocab) float32 -> logprobs for the token after the prompt"""
        lg = lg + self.first
        return lg - mx.logsumexp(lg, axis=-1, keepdims=True)

    def __call__(self, lg, prev1, prev2, last_ts, seen):
        """lg (rows, vocab) float32, prev1/prev2 each row's last two tokens, last_ts the last timestamp it emitted,
        seen what it generated so far (or None without a repetition penalty) -> logprobs"""
        ninf, ts_begin = float('-inf'), self.ts_begin
        lg = lg + self.base
        if seen is not None: # discourage what was already said, sign aware so it always lowers the logit
            p = self.repetition_penalty
            lg = mx.where(seen, mx.where(lg > 0, lg / p, lg * p), lg)

        # timestamps come in pairs and can only move forward
        last_was_ts, penultimate_was_ts = (prev1 >= ts_begin)[:, None], (prev2 >= ts_begin)[:, None]
        closed, opened = last_was_ts & penultimate_was_ts, last_was_ts & ~penultimate_was_ts
        lg = mx.where(closed & self.is_ts, ninf, lg)         # the pair is closed, has to be text
        lg = mx.where(opened & self.is_before_eot, ninf, lg) # the pair is open, has to be closed
        lg = mx.where(self.is_ts & (self.idx < last_ts[:, None] + mx.where(opened, 0, 1)), ninf, lg) # no going backwards

        # if the timestamps are collectively likelier than any text token, take a timestamp
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        force = mx.logsumexp(lp[:, ts_begin:], axis=-1, keepdims=True) > mx.max(lp[:, :ts_begin], axis=-1, keepdims=True)
        lg = mx.where(force & self.is_text, ninf, lg)
        return lg - mx.logsumexp(lg, axis=-1, keepdims=True)

    def advance(self, chosen, prev1, last_ts, seen):
        """the per row state after appending chosen, -> (prev1, prev2, last_ts, seen)"""
        last_ts = mx.where(chosen >= self.ts_begin, chosen, last_ts)
        if seen is not None: seen = seen | (self.idx[None, :] == chosen[:, None])
        return chosen, prev1, last_ts, seen

    def initial_seen(self, rows):
        return mx.zeros((rows, self.idx.shape[0]), dtype=mx.bool_) if self.repetition_penalty != 1 else None


def sample(decoder, rules, logits, kv_cache, xa, temperatures, sampling_topk, max_steps):
    """greedy and sampled rows in one batch, one temperature per row (0 is greedy)

    -> (sequences (rows, length) with everything from the eot on still in it, sum of logprobs per row)"""
    eot, ninf = rules.eot, float('-inf')
    rows = len(temperatures)
    sampled = temperatures != 0
    # the greedy rows still go through the divide, so keep it at 1 for them rather than dividing by zero
    divisor = mx.array(np.where(sampled, temperatures, 1).reshape(rows, 1).astype(np.float32))
    is_greedy = mx.array(~sampled)
    any_sampled = bool(sampled.any())

    def select(lp):
        chosen = mx.argmax(lp, axis=-1)
        if any_sampled: # one extra argmax is cheaper than splitting the batch by temperature
            scaled = lp / divisor
            if sampling_topk:
                kth = mx.min(mx.topk(scaled, sampling_topk, axis=-1), axis=-1, keepdims=True)
                scaled = mx.where(scaled < kth, ninf, scaled)
            chosen = mx.where(is_greedy, chosen, mx.random.categorical(scaled, axis=-1))
        chosen = chosen.astype(mx.int32)
        return chosen, mx.take_along_axis(lp, chosen[:, None], axis=-1)[:, 0]

    chosen, sum_logprobs = select(rules.first_step(logits[:, -1].astype(mx.float32)))
    alive = mx.ones((rows,), dtype=mx.bool_) # the first token is a timestamp, so nothing has ended yet
    prev1, prev2, last_ts, seen = rules.advance(chosen, mx.full((rows,), rules.ts_begin, dtype=mx.int32), chosen,
                                                rules.initial_seen(rows))
    out = [chosen]

    for step in range(1, max_steps):
        logits, kv_cache, _ = decoder(chosen[:, None], xa, kv_cache=kv_cache)
        chosen, step_logprob = select(rules(logits[:, -1].astype(mx.float32), prev1, prev2, last_ts, seen))

        sum_logprobs = sum_logprobs + mx.where(alive, step_logprob, 0)
        chosen = mx.where(alive, chosen, eot)
        alive = alive & (chosen != eot)
        prev1, prev2, last_ts, seen = rules.advance(chosen, prev1, last_ts, seen)
        out.append(chosen)

        mx.async_eval(chosen, alive, sum_logprobs, last_ts, kv_cache.state) # bound the graph without waiting on it
        if step % _SYNC_EVERY == 0 and not alive.any().item(): break

    return np.array(mx.stack(out, axis=1)), np.array(sum_logprobs)

