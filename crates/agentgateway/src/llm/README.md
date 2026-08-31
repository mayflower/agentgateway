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

`responseCache` on the AI policy stores an upstream response and serves it again for an identical
finalized request, replacing the provider call. Unrelated to `promptCaching`, which annotates a
request so that the *provider* caches part of the prompt.

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

The gateway always keys on the finalized body, resolved provider, model, client format, upstream
target and path, the forwarded headers that change what the provider returns, and the credential it
attached. Per-request values such as `traceparent` and SigV4 signatures are excluded, since a key
containing them would never match twice; a request whose credential is itself per-request bypasses
instead.

The `key` expressions are what stop one caller receiving another's completion. The gateway cannot
infer which dimension identifies a tenant, so a `responseCache` without them serves no hits.

Eligibility is non-streaming chat completions with a successful response and a `content-length`
within the route's buffer limit. Everything else bypasses unchanged.

A hit returns a byte-identical completion, so callers relying on sampling variation will not get it.
It re-enters the normal buffered path, so translation, guards, usage extraction, and logging behave
as on a miss, and it is not counted or charged as a provider call. Access log lines carry
`agw.ai.response_cache: hit`.

The cache is in-process and per policy instance: not shared between replicas, lost on restart,
dropped when a reload replaces the policy. Concurrent identical requests all reach the provider.
