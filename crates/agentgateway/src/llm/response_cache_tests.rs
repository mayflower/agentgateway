//! Exact response cache: canonicalization, key completeness, key stability, and replay hygiene.

use std::sync::Arc;

use ::http::StatusCode;

use super::*;
use crate::llm::{InputFormat, LLMRequest};
use crate::types::agent::Target;

fn expr(src: &str) -> Arc<cel::Expression> {
	Arc::new(cel::Expression::new_strict(src).expect("expression"))
}

/// A cache keyed on the caller's authorization, which is what a real deployment must declare.
fn cache() -> Arc<ResponseCache> {
	Arc::new(ResponseCache::new(ResponseCacheConfig {
		key: vec![expr(r#"request.headers["authorization"]"#)],
		ttl: expr(r#"duration("300s")"#),
		max_entries: 16,
	}))
}

fn request(headers: &[(&str, &str)]) -> crate::http::Request {
	let mut builder = ::http::Request::builder().uri("http://api.example.com/v1/chat/completions");
	for (name, value) in headers {
		builder = builder.header(*name, *value);
	}
	builder
		.header("authorization", "Bearer caller-a")
		.body(crate::http::Body::empty())
		.expect("request")
}

const BODY: &str = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;

/// Build a key from a body and a request, using the same field set `prepare` uses.
fn key_of(cache: &ResponseCache, body: &str, req: &crate::http::Request) -> CacheKey {
	let canonical = canonicalize(body.as_bytes()).expect("canonical");
	let uri = req.uri().clone();
	let inputs = CacheKeyInputs {
		canonical_body: &canonical,
		provider: "openai",
		model: "gpt-4o",
		input_format: "Completions",
		target: "api.example.com:443",
		path_and_query: uri.path_and_query().map(|p| p.as_str()).unwrap_or("/"),
		headers: req.headers(),
	};
	cache.key(&inputs, req).expect("key")
}

// ── canonicalization ────────────────────────────────────────────────────────

#[test]
fn object_key_order_is_normalized_recursively() {
	let a = canonicalize(br#"{"b":1,"a":{"d":2,"c":3}}"#).expect("a");
	let b = canonicalize(br#"{"a":{"c":3,"d":2},"b":1}"#).expect("b");
	assert_eq!(a, b);
	assert_eq!(a, Bytes::from_static(br#"{"a":{"c":3,"d":2},"b":1}"#));
}

#[test]
fn array_order_and_values_are_preserved() {
	let a = canonicalize(br#"{"x":[1,2,3]}"#).expect("a");
	let b = canonicalize(br#"{"x":[3,2,1]}"#).expect("b");
	assert_ne!(a, b, "array order carries meaning and must not be sorted");
	assert_eq!(a, Bytes::from_static(br#"{"x":[1,2,3]}"#));
}

#[test]
fn prompt_text_is_left_exactly_alone() {
	let canonical = canonicalize(br#"{"m":"  Hello   World  "}"#).expect("canonical");
	assert_eq!(
		canonical,
		Bytes::from_static(br#"{"m":"  Hello   World  "}"#)
	);
}

#[test]
fn non_json_body_does_not_canonicalize() {
	assert!(canonicalize(b"not json").is_none());
}

// ── key completeness ────────────────────────────────────────────────────────

#[test]
fn reordered_body_keys_hit_the_same_entry() {
	let cache = cache();
	let req = request(&[]);
	let a = key_of(
		&cache,
		r#"{"model":"gpt-4o","messages":[],"temperature":0.5}"#,
		&req,
	);
	let b = key_of(
		&cache,
		r#"{"temperature":0.5,"messages":[],"model":"gpt-4o"}"#,
		&req,
	);
	assert_eq!(a, b);
}

#[test]
fn a_changed_value_misses() {
	let cache = cache();
	let req = request(&[]);
	let a = key_of(&cache, r#"{"temperature":0.5}"#, &req);
	let b = key_of(&cache, r#"{"temperature":0.6}"#, &req);
	assert_ne!(a, b);
}

#[test]
fn a_different_caller_misses() {
	let cache = cache();
	let a = key_of(&cache, BODY, &request(&[]));

	let other = ::http::Request::builder()
		.uri("http://api.example.com/v1/chat/completions")
		.header("authorization", "Bearer caller-b")
		.body(crate::http::Body::empty())
		.expect("request");
	let b = key_of(&cache, BODY, &other);

	assert_ne!(a, b, "a response must not be shared across callers");
}

#[test]
fn a_different_model_or_provider_or_target_misses() {
	let cache = cache();
	let req = request(&[]);
	let canonical = canonicalize(BODY.as_bytes()).expect("canonical");
	let base = CacheKeyInputs {
		canonical_body: &canonical,
		provider: "openai",
		model: "gpt-4o",
		input_format: "Completions",
		target: "api.example.com:443",
		path_and_query: "/v1/chat/completions",
		headers: req.headers(),
	};
	let baseline = cache.key(&base, &req).expect("key");

	for changed in [
		CacheKeyInputs {
			model: "gpt-4o-mini",
			..CacheKeyInputs { ..base }
		},
		CacheKeyInputs {
			provider: "azure",
			..CacheKeyInputs { ..base }
		},
		CacheKeyInputs {
			target: "other.example.com:443",
			..CacheKeyInputs { ..base }
		},
		CacheKeyInputs {
			path_and_query: "/v1/chat/completions?beta=1",
			..CacheKeyInputs { ..base }
		},
		CacheKeyInputs {
			input_format: "Messages",
			..CacheKeyInputs { ..base }
		},
	] {
		assert_ne!(
			baseline,
			cache.key(&changed, &req).expect("key"),
			"every resolved execution dimension must separate entries"
		);
	}
}

#[test]
fn a_behavior_changing_forwarded_header_misses() {
	// `anthropic-beta` selects provider behavior and never reaches the body on the native path.
	let cache = cache();
	let plain = key_of(&cache, BODY, &request(&[]));
	let beta = key_of(
		&cache,
		BODY,
		&request(&[("anthropic-beta", "computer-use-2025-01-24")]),
	);
	assert_ne!(plain, beta);
}

// ── key stability ───────────────────────────────────────────────────────────

#[test]
fn volatile_headers_do_not_poison_the_key() {
	// A key including the per-request signing headers or the trace context would never hit.
	let cache = cache();
	let first = key_of(
		&cache,
		BODY,
		&request(&[
			("traceparent", "00-aaaaaaaaaaaa-bbbbbbbb-01"),
			("x-amz-date", "20260830T101500Z"),
			("authorization-signature", "AWS4-HMAC-SHA256 Signature=aaa"),
		]),
	);
	let second = key_of(
		&cache,
		BODY,
		&request(&[
			("traceparent", "00-cccccccccccc-dddddddd-01"),
			("x-amz-date", "20260830T101600Z"),
			("authorization-signature", "AWS4-HMAC-SHA256 Signature=bbb"),
		]),
	);
	assert_eq!(
		first, second,
		"per-request headers must not take part in the key"
	);
}

// ── configuration guards ────────────────────────────────────────────────────

#[test]
fn a_cache_without_declared_key_expressions_serves_nothing() {
	let cache = ResponseCache::new(ResponseCacheConfig {
		key: vec![],
		ttl: expr(r#"duration("300s")"#),
		max_entries: 16,
	});
	assert!(
		!cache.serves_hits(),
		"sharing across every caller must be refused, not done silently"
	);
}

#[test]
fn ttl_accepts_a_literal_duration_and_rejects_a_non_duration() {
	let req = request(&[]);
	assert_eq!(cache().ttl(&req), Some(Duration::from_secs(300)));

	let bad = ResponseCache::new(ResponseCacheConfig {
		key: vec![expr("request.path")],
		ttl: expr("request.path"),
		max_entries: 16,
	});
	assert_eq!(bad.ttl(&req), None, "a non-duration TTL disables storage");
}

// ── storage, expiry, replay ─────────────────────────────────────────────────

fn upstream_headers() -> HeaderMap {
	let mut headers = HeaderMap::new();
	headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
	headers.insert(
		header::DATE,
		"Sat, 30 Aug 2026 10:15:00 GMT".parse().unwrap(),
	);
	headers.insert("x-request-id", "req_upstream_1".parse().unwrap());
	headers.insert(
		"anthropic-ratelimit-requests-remaining",
		"42".parse().unwrap(),
	);
	headers.insert(header::SET_COOKIE, "session=secret".parse().unwrap());
	headers
}

#[test]
fn a_stored_entry_is_returned_until_it_expires() {
	let cache = cache();
	let key = key_of(&cache, BODY, &request(&[]));

	cache.insert(
		key,
		CachedResponse::capture(
			StatusCode::OK,
			&upstream_headers(),
			Bytes::from_static(b"{}"),
			Duration::from_secs(300),
		),
	);
	assert!(cache.lookup(&key).is_some());

	cache.insert(
		key,
		CachedResponse::capture(
			StatusCode::OK,
			&upstream_headers(),
			Bytes::from_static(b"{}"),
			Duration::ZERO,
		),
	);
	assert!(cache.lookup(&key).is_none(), "an expired entry is a miss");
}

#[test]
fn replay_carries_only_headers_that_describe_the_stored_body() {
	let cached = CachedResponse::capture(
		StatusCode::OK,
		&upstream_headers(),
		Bytes::from_static(b"{}"),
		Duration::from_secs(300),
	);
	let resp = cached.into_response();

	assert_eq!(resp.status(), StatusCode::OK);
	assert_eq!(
		resp.headers().get(header::CONTENT_TYPE).unwrap(),
		"application/json"
	);
	for leaked in [
		header::DATE.as_str(),
		"x-request-id",
		"anthropic-ratelimit-requests-remaining",
		header::SET_COOKIE.as_str(),
	] {
		assert!(
			resp.headers().get(leaked).is_none(),
			"{leaked} describes another client's exchange and must not be replayed"
		);
	}
}

#[test]
fn a_key_is_never_printed_in_full() {
	let cache = cache();
	let key = key_of(&cache, BODY, &request(&[]));
	let rendered = format!("{key:?}");
	assert!(rendered.starts_with("CacheKey("), "{rendered}");
	assert!(
		rendered.len() < 24,
		"debug output must not publish key material: {rendered}"
	);
}

// ── eligibility ─────────────────────────────────────────────────────────────

fn llm_request(input_format: InputFormat, streaming: bool) -> LLMRequest {
	LLMRequest {
		input_format,
		cache_convention: crate::llm::CacheTokenConvention::pending(),
		request_model: strng::literal!("gpt-4o"),
		streaming,
		provider: strng::literal!("openai"),
		input_tokens: None,
		params: Default::default(),
		prompt: Default::default(),
		provider_state: None,
	}
}

fn policy_with_cache() -> crate::llm::Policy {
	crate::llm::Policy {
		response_cache: Some(cache()),
		..Default::default()
	}
}

fn eligible_request() -> crate::http::Request {
	let mut req = request(&[]);
	let canonical = canonicalize(BODY.as_bytes()).expect("canonical");
	req.extensions_mut().insert(CanonicalRequestBody(canonical));
	req
}

#[test]
fn a_configured_completions_request_is_prepared() {
	let target = Target::Hostname(strng::literal!("api.example.com"), 443);
	let prepared = prepare(
		Some(&policy_with_cache()),
		Some(&llm_request(InputFormat::Completions, false)),
		&target,
		&eligible_request(),
	);
	assert!(prepared.is_some());
}

#[test]
fn ineligible_requests_bypass() {
	let target = Target::Hostname(strng::literal!("api.example.com"), 443);
	let policy = policy_with_cache();

	// Streaming, other route shapes, no policy, and a request that never rendered a canonical body.
	assert!(
		prepare(
			Some(&policy),
			Some(&llm_request(InputFormat::Completions, true)),
			&target,
			&eligible_request()
		)
		.is_none(),
		"streaming bypasses"
	);
	assert!(
		prepare(
			Some(&policy),
			Some(&llm_request(InputFormat::Messages, false)),
			&target,
			&eligible_request()
		)
		.is_none(),
		"only chat completions are eligible in this version"
	);
	assert!(
		prepare(
			Some(&crate::llm::Policy::default()),
			Some(&llm_request(InputFormat::Completions, false)),
			&target,
			&eligible_request()
		)
		.is_none(),
		"no configuration means no cache"
	);
	assert!(
		prepare(
			Some(&policy),
			Some(&llm_request(InputFormat::Completions, false)),
			&target,
			&request(&[])
		)
		.is_none(),
		"without a canonical body there is nothing safe to key on"
	);
}

#[test]
fn a_different_accept_encoding_misses() {
	// The stored bytes carry the upstream's encoding, so replaying them to a client that asked for a
	// different encoding would hand back something it cannot read.
	let cache = cache();
	let identity = key_of(&cache, BODY, &request(&[]));
	let gzip = key_of(&cache, BODY, &request(&[("accept-encoding", "gzip")]));
	assert_ne!(identity, gzip);
}

#[test]
fn a_different_upstream_credential_misses() {
	// Two routes can share one backend policy, and therefore one cache, while attaching different
	// credentials. Nothing in the operator's key expressions necessarily notices that.
	let cache = cache();
	let with_key = |api_key: &str| {
		let req = ::http::Request::builder()
			.uri("http://api.example.com/v1/chat/completions")
			.header("authorization", "Bearer caller-a")
			.header("x-api-key", api_key)
			.body(crate::http::Body::empty())
			.expect("request");
		key_of(&cache, BODY, &req)
	};
	assert_ne!(with_key("sk-tenant-one"), with_key("sk-tenant-two"));
}

#[test]
fn a_per_request_signature_bypasses_instead_of_keying_on_itself() {
	// SigV4 rebuilds the header every request, so it can neither be keyed on nor ignored.
	let cache = cache();
	let req = ::http::Request::builder()
		.uri("http://api.example.com/v1/chat/completions")
		.header(
			"authorization",
			"AWS4-HMAC-SHA256 Credential=AKIA/20260830/us-east-1/bedrock/aws4_request, Signature=abc",
		)
		.body(crate::http::Body::empty())
		.expect("request");
	let canonical = canonicalize(BODY.as_bytes()).expect("canonical");
	let inputs = CacheKeyInputs {
		canonical_body: &canonical,
		provider: "bedrock",
		model: "claude",
		input_format: "Completions",
		target: "bedrock.us-east-1.amazonaws.com:443",
		path_and_query: "/model/claude/converse",
		headers: req.headers(),
	};
	assert!(
		cache.key(&inputs, &req).is_none(),
		"a signature that changes every request must bypass, not produce a key that never matches"
	);
}
