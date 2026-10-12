"""Budget a final-answer turn without rewriting any cached conversation tokens."""

import json

FINAL_ANSWER_PROMPT = (
    "Stop searching now. Do not call any more tools. Using only the evidence "
    "already available in this conversation, provide your best final answer. "
    "Keep your remaining reasoning concise and use this format:\n"
    "Explanation: your explanation, citing evidence document IDs in square brackets\n"
    "Exact Answer: your succinct best answer\n"
    "Confidence: your confidence from 0% to 100%"
)


class ContextLimitError(ValueError):
    """No room remains for a generation request within the configured context."""


class ContextPolicy:
    def __init__(self, session, *, final_answer_tokens=32768, context_margin=32768):
        if final_answer_tokens < 1 or context_margin < final_answer_tokens:
            raise ValueError(
                "Context margin must cover the positive final-answer reserve"
            )
        self.session = session
        self.final_answer_tokens = final_answer_tokens
        self.context_margin = context_margin
        self.reason = None
        self.prompted = False
        self.omitted_results = 0
        self.trimming = []
        self.event = None

    @staticmethod
    def final_message():
        return {"role": "user", "content": FINAL_ANSWER_PROMPT}

    def prepare_turn(self, messages, tools, turn_max_tokens):
        final_available = self.session.remaining_context(
            messages + [self.final_message()], tools
        )
        # If every new tool result was omitted, there are no new messages to
        # render yet. Budget the final instruction first in that case.
        available = (
            final_available
            if self.reason is not None
            else self.session.remaining_context(messages, tools)
        )
        if self.reason is None:
            if available <= self.context_margin or final_available < turn_max_tokens:
                self.reason = "context_limit_approaching"
        if self.reason is not None and not self.prompted:
            if final_available < 1:
                raise ContextLimitError(
                    "No context space remains for the final-answer request"
                )
            messages.append(self.final_message())
            self.prompted = True
            available = final_available
            self.event = {
                "reason": self.reason,
                "prompt_tokens": self.session.factory.context_length - available - 1,
                "remaining_context_tokens": available,
                "max_tokens_per_turn": turn_max_tokens,
            }
        # The final-answer reserve protects space from new tool results; it is
        # not a separate output cap. Every turn receives the same allowance.
        maximum = min(turn_max_tokens, available)
        if maximum < 1:
            raise ContextLimitError("No context space remains for generation")
        return maximum

    def tool_result_fits(self, messages, tools, tool_message, turn_max_tokens):
        available = self.session.remaining_context(
            messages + [tool_message, self.final_message()], tools
        )
        return available >= min(turn_max_tokens, self.final_answer_tokens)

    def fit_search_results(self, messages, tools, call, hits, turn_max_tokens):
        """Keep a ranked prefix of results and, if possible, part of the next.

        Count the fully serialized message with Horizon's tokenizer, including
        document IDs, JSON, message boundaries and the final-answer instruction.
        Only new results are shortened; previously cached tokens are untouched.
        """

        def message_for(results):
            return {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": json.dumps(results, ensure_ascii=False, indent=2),
            }

        def fits(results):
            return self.tool_result_fits(
                messages, tools, message_for(results), turn_max_tokens
            )

        full_message = message_for(hits)
        if self.tool_result_fits(messages, tools, full_message, turn_max_tokens):
            return full_message, hits

        self.reason = "context_limit_approaching"
        kept = []
        truncated = None
        for hit in hits:
            if fits(kept + [hit]):
                kept.append(hit)
                continue
            snippet = hit["snippet"]
            best = None
            low, high = 1, len(snippet) - 1
            while low <= high:
                length = (low + high) // 2
                candidate = {**hit, "snippet": snippet[:length], "truncated": True}
                if fits(kept + [candidate]):
                    best = candidate
                    low = length + 1
                else:
                    high = length - 1
            # Token counts need not be strictly monotonic with prefix length.
            # Retain only a candidate that actually passed the full fit check.
            if best is not None:
                kept.append(best)
                truncated = {
                    "docid": hit["docid"],
                    "original_chars": len(snippet),
                    "retained_chars": len(best["snippet"]),
                }
            break
        self.trimming.append(
            {
                "tool_call_id": call.id,
                "original_hits": len(hits),
                "retained_hits": len(kept),
                "omitted_hits": len(hits) - len(kept),
                "truncated_snippet": truncated,
            }
        )
        if not kept:
            return self.stop_search_message(call), []
        return message_for(kept), kept

    def stop_search_message(self, call):
        self.reason = self.reason or "context_limit_approaching"
        self.omitted_results += 1
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.function.name,
            "content": json.dumps(
                {
                    "error": "Search stopped to reserve context for the final answer. "
                    "Use the evidence already available; do not request more searches."
                }
            ),
        }

    def summary(self):
        return {
            "requested": self.prompted,
            "event": self.event,
            "omitted_tool_results": self.omitted_results,
            "trimmed_tool_results": sum(t["retained_hits"] > 0 for t in self.trimming),
            "search_result_trimming": self.trimming,
        }
