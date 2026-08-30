# Agentgateway LLM Functionality

This module builds functionality for handling LLM requests.
This includes support for multiple different types of requests (OpenAI completions, Embeddings, Anthropic messages, etc),
policy and manipulation of these, parsing, and in some cases conversion.

In order to facilitate maximum compatibility (across providers or across versions, as new fields are added),
we use a "passthrough" approach to parsing. Each message includes a final `rest` field that stores all unknown fields:
```rust
#[serde(flatten, default)]
pub rest: serde_json::Value
```
Only fields we specifically operate on (like `model`) need to be included in the type definitions.

However, in some cases having the full typed definitions is useful, such as for conversion from one type to another.
In these, we have additional `typed` variation that we upgrade the passhthrough type to internally.
## Exact response cache

`responseCache` on the AI policy is a gateway-side cache of upstream responses. It is unrelated to
`promptCaching`, which annotates requests so that a *provider* caches part of the prompt. This one
stores the provider's reply and serves it again, replacing the upstream call entirely.

```yaml
policies:
  ai:
    responseCache:
      # Dimensions only the deployment knows. Without at least one, no hit is ever served.
      key:
        - request.headers["authorization"]
      ttl: 5m
      maxEntries: 1024
```

**What the key covers.** The gateway always keys on the finalized request body, the resolved
provider, model, client-facing format, upstream target, and upstream path, plus the forwarded
headers that change what the provider returns (`anthropic-beta`, the OpenAI organization and project
headers), `accept-encoding`, which decides how the stored bytes are encoded, and the upstream
credential the gateway attached. Per-request values — `x-amz-date`, `traceparent` — are deliberately
excluded; a key containing them would never match twice.

A credential that is rebuilt for every request cannot be keyed on and must not be ignored, so those
requests bypass the cache instead. That currently means AWS SigV4: extracting the stable credential
scope out of a signature is possible but not yet worth the parsing.

The `key` expressions are what stop one caller receiving another caller's completion. The gateway
cannot infer which dimension identifies a tenant in a given deployment, so it does not guess: a
`responseCache` with no `key` expressions is accepted but serves no hits.

**Canonicalization.** Object keys are ordered recursively before hashing. Arrays, values, and prompt
text are untouched. Whitespace and the ordering of typed fields never reach the key at all — the
request was parsed and re-serialized before this point — so the ordering pass only affects values
carried through as raw JSON, such as the flattened `rest` maps and `tools`.

**Eligibility.** Chat completions, non-streaming, successful responses only. Anything else bypasses
and behaves exactly as it does today. Responses without a usable `content-length`, or larger than the
route's response buffer limit, are not stored: collecting them here could fail a request that would
otherwise have succeeded.

**Determinism.** A hit returns a byte-identical completion. With `temperature` above zero, callers
relying on sampling variation will not get it. The policy is opt-in per route precisely so that this
is a deliberate choice.

**Replay.** The stored entry is the upstream response as it arrived, so a hit re-enters the normal
buffered path: response translation, response guards, usage extraction, and access logging all run as
they would on a miss. Only `content-type` and `content-encoding` are replayed from the stored
headers; `date`, `x-request-id`, provider rate-limit headers, and `set-cookie` describe an exchange
the current client did not make and are dropped.

**Accounting.** A hit is not a provider call: no outbound span, no `upstream_call_duration`
observation, no upstream duration, and no second charge to `gen_ai_client_cost`. Token usage *is*
still reported and the token rate-limit reservation *is* still reconciled — the client consumed those
tokens, and leaving the admission reservation unsettled would over-charge it. Access log lines for a
hit carry `agw.ai.response_cache: hit`.

**Scope.** In-process and per policy instance: entries are not shared between replicas, are lost on
restart, and are dropped when a configuration reload replaces the policy. There is no invalidation
API. Concurrent identical requests all reach the provider; only the first of them stores.
