//! Gateway-side exact response cache for buffered chat completions.
//!
//! A hit replaces the upstream provider call with a previously stored response for a request whose
//! finalized syntax and resolved execution context are identical. This is not provider-side prompt
//! caching: no `cache_control`, `cachePoint`, or `prompt_cache_breakpoint` marker is involved, and
//! nothing here changes what is sent upstream on a miss.
//!
//! Shape follows the two caches already in the tree. The operator-declared CEL `key` and the CEL
//! `ttl` mirror `http::ext_authz::CacheConfig`; the length-delimited SHA-256 digest mirrors the
//! OAuth token cache, so no raw prompt, header, or credential is retained as key material.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ::http::{HeaderMap, HeaderName, StatusCode, header};
use bytes::Bytes;
use quick_cache::sync::Cache;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::crypto::digest::Sha256;
use crate::*;

/// Entries kept per configured cache when the operator does not choose.
const DEFAULT_MAX_ENTRIES: usize = 1024;

fn default_max_entries() -> usize {
	DEFAULT_MAX_ENTRIES
}

/// Request headers that change the upstream result and are not visible in the request body.
///
/// `anthropic-beta` is the load-bearing case: on the native Anthropic path it is forwarded verbatim
/// and selects provider behavior, while the Bedrock conversion folds it into the body instead. A key
/// built from the body alone is therefore complete for one and unsafe for the other.
///
/// Deliberately *not* here: `anthropic-version` is set by the gateway to a constant, and Azure's
/// `api-version` lives in the query string, which the key already covers.
///
/// `accept-encoding` is here for a different reason. It does not change what the provider generates,
/// but agentgateway forwards it untouched, so it decides how the stored bytes are encoded. Without
/// it a gzip response cached for one client would be replayed to a client that never asked for gzip.
const KEYED_REQUEST_HEADERS: &[&str] = &[
	"anthropic-beta",
	"openai-organization",
	"openai-project",
	"accept-encoding",
];

/// Response headers replayed from a stored entry.
///
/// An allowlist, not a denylist: everything else an upstream sends is specific to the exchange that
/// produced it — `date`, `x-request-id`, `set-cookie`, and the provider rate-limit headers all
/// describe a request the current client did not make. `content-encoding` is here because it
/// describes the stored bytes themselves, not the exchange.
const REPLAYED_RESPONSE_HEADERS: &[&str] = &["content-type", "content-encoding"];

#[apply(schema!)]
pub struct ResponseCacheConfig {
	/// CEL expressions contributing additional dimensions to the cache key, evaluated against the
	/// request. The gateway always keys on the finalized request body, the resolved provider, model,
	/// client-facing format, upstream target, and upstream path; these expressions add what only the
	/// deployment knows, such as the caller identity a response must not be shared across.
	///
	/// Leaving this empty shares responses between every caller reaching the same route with the same
	/// body. That is refused rather than done silently: with no expressions the cache serves no hits.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub key: Vec<Arc<cel::Expression>>,
	/// How long a stored response may be reused. Accepts a duration literal such as `5m`, or a CEL
	/// expression returning a duration.
	#[serde(deserialize_with = "crate::cel::de_duration_or_expression")]
	pub ttl: Arc<cel::Expression>,
	/// Maximum number of stored responses. `0` disables storage without removing the policy.
	///
	/// This bounds the entry count, not bytes. Each entry is at most one buffered response, so the
	/// worst case is `maxEntries` multiplied by the response buffer limit already in force for the
	/// route. The cache is per-process and lost on restart.
	#[serde(default = "default_max_entries")]
	pub max_entries: usize,
}

/// A configured cache: the policy settings plus the storage they describe.
///
/// Held behind an `Arc` on the LLM policy so that per-request policy merging shares one store rather
/// than building a new one, and so that a configuration reload constructs a new policy with a new,
/// empty store. That is the entire invalidation story for a configuration change; there is no
/// imperative invalidation API, matching the outcome of #1956.
pub struct ResponseCache {
	config: ResponseCacheConfig,
	store: Cache<CacheKey, CachedResponse>,
}

impl ResponseCache {
	pub fn new(config: ResponseCacheConfig) -> Self {
		// `quick_cache` panics on a zero-capacity cache; a zero here means the operator disabled
		// storage, which is expressed as a cache that never retains anything.
		let capacity = config.max_entries.max(1);
		Self {
			store: Cache::new(capacity),
			config,
		}
	}

	pub fn config(&self) -> &ResponseCacheConfig {
		&self.config
	}

	/// CEL expressions this policy references, for expression registration.
	pub fn expressions(&self) -> impl Iterator<Item = &cel::Expression> {
		self
			.config
			.key
			.iter()
			.map(Arc::as_ref)
			.chain(std::iter::once(self.config.ttl.as_ref()))
	}

	/// Whether the configuration can serve hits at all. An empty key list cannot express caller
	/// identity, so serving a hit would share one caller's completion with another.
	pub fn serves_hits(&self) -> bool {
		!self.config.key.is_empty() && self.config.max_entries > 0
	}

	pub fn lookup(&self, key: &CacheKey) -> Option<CachedResponse> {
		let cached = self.store.get(key)?;
		let now = Instant::now();
		if cached.expires_at <= now {
			self.store.remove_if(key, |c| c.expires_at <= now);
			return None;
		}
		Some(cached)
	}

	pub fn insert(&self, key: CacheKey, response: CachedResponse) {
		self.store.insert(key, response);
	}

	/// Evaluate the configured TTL. A TTL that does not evaluate to a positive duration disables
	/// storage for this request rather than falling back to a default the operator did not choose.
	pub fn ttl(&self, req: &crate::http::Request) -> Option<Duration> {
		let exec = cel::Executor::new_request(req);
		let value = exec.eval(&self.config.ttl).ok()?;
		let cel::Value::Duration(ttl) = value else {
			return None;
		};
		let ttl = ttl.to_std().ok()?;
		(!ttl.is_zero()).then_some(ttl)
	}

	/// Build the key for a request, or `None` when a required dimension cannot be represented. A
	/// missing dimension bypasses the cache; it never produces a partial key.
	pub fn key(&self, inputs: &CacheKeyInputs<'_>, req: &crate::http::Request) -> Option<CacheKey> {
		let mut digest = CacheKeyDigest::new();

		// Gateway-computed dimensions the operator cannot omit.
		digest.field(inputs.canonical_body);
		digest.field(inputs.provider.as_bytes());
		digest.field(inputs.model.as_bytes());
		digest.field(inputs.input_format.as_bytes());
		digest.field(inputs.target.as_bytes());
		digest.field(inputs.path_and_query.as_bytes());

		// Forwarded headers that change provider behavior. Iterated over a fixed list so the digest
		// does not depend on header map ordering, and so a header nobody has allowlisted can never
		// silently join the key.
		for name in KEYED_REQUEST_HEADERS {
			digest.field(name.as_bytes());
			for value in inputs.headers.get_all(*name) {
				digest.field(value.as_bytes());
			}
			digest.field(b"\0");
		}

		// Operator-declared dimensions.
		let exec = cel::Executor::new_request(req);
		for expression in &self.config.key {
			let value = exec.eval(expression).ok()?;
			let bytes = cel::value_as_byte_or_json(value).ok()?;
			digest.field(&bytes);
		}

		Some(digest.finish())
	}
}

impl std::fmt::Debug for ResponseCache {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ResponseCache")
			.field("config", &self.config)
			.field("len", &self.store.len())
			.finish()
	}
}

impl Serialize for ResponseCache {
	fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		self.config.serialize(serializer)
	}
}

impl<'de> Deserialize<'de> for ResponseCache {
	fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
		Ok(Self::new(ResponseCacheConfig::deserialize(deserializer)?))
	}
}

/// Gateway-computed key dimensions, gathered where the finalized request exists.
pub struct CacheKeyInputs<'a> {
	pub canonical_body: &'a [u8],
	pub provider: &'a str,
	pub model: &'a str,
	pub input_format: &'a str,
	/// Resolved upstream target. The upstream wire format is not a separate field: it is determined
	/// by the provider and the provider-specific path, both of which are already here.
	pub target: &'a str,
	pub path_and_query: &'a str,
	pub headers: &'a HeaderMap,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey([u8; 32]);

// Only a short prefix, so a debug line can correlate two requests without publishing key material.
impl std::fmt::Debug for CacheKey {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"CacheKey({:02x}{:02x}{:02x}{:02x}..)",
			self.0[0], self.0[1], self.0[2], self.0[3]
		)
	}
}

/// Length-delimited so that field boundaries cannot be forged by concatenating adjacent values.
struct CacheKeyDigest(Sha256);

impl CacheKeyDigest {
	fn new() -> Self {
		Self(Sha256::new())
	}

	fn field(&mut self, bytes: impl AsRef<[u8]>) {
		let bytes = bytes.as_ref();
		self.0.update((bytes.len() as u64).to_le_bytes());
		self.0.update(bytes);
	}

	fn finish(self) -> CacheKey {
		CacheKey(self.0.finalize())
	}
}

/// A stored upstream response, held in the shape it arrived in so that replay re-enters the normal
/// buffered response path — translation, response guards, usage accounting, and logging all run on a
/// hit exactly as they would on a miss.
#[derive(Clone, Debug)]
pub struct CachedResponse {
	status: StatusCode,
	headers: HeaderMap,
	body: Bytes,
	expires_at: Instant,
}

impl CachedResponse {
	/// Capture a response for storage, keeping only the allowlisted headers.
	pub fn capture(status: StatusCode, headers: &HeaderMap, body: Bytes, ttl: Duration) -> Self {
		let mut kept = HeaderMap::new();
		for name in REPLAYED_RESPONSE_HEADERS {
			let name = HeaderName::from_static(name);
			for value in headers.get_all(&name) {
				kept.append(name.clone(), value.clone());
			}
		}
		Self {
			status,
			headers: kept,
			body,
			expires_at: Instant::now() + ttl,
		}
	}

	pub fn into_response(self) -> crate::http::Response {
		let mut builder = ::http::Response::builder().status(self.status);
		if let Some(headers) = builder.headers_mut() {
			headers.clone_from(&self.headers);
			// Describes the body actually being sent, not the one the upstream sent originally.
			headers.remove(header::CONTENT_LENGTH);
		}
		builder
			.body(crate::http::Body::from(self.body))
			.expect("cached response parts were validated when the response was captured")
	}
}

/// The finalized upstream request body, carried from rendering to the point where the key is built.
///
/// Canonicalized once at insertion into the request rather than at key time: it is the same work
/// either way, and doing it here keeps JSON handling in the LLM layer.
#[derive(Clone, Debug)]
pub struct CanonicalRequestBody(pub Bytes);

/// Recursively order object keys, leaving arrays, values, and prompt text untouched.
///
/// This is the whole canonicalization surface. Whitespace and typed-field ordering are already
/// normalized upstream of here: the request was parsed into typed structs and re-serialized with
/// `serde_json::to_vec`, so only values carried through as raw JSON — the flattened `rest` maps and
/// fields like `tools` or `tool_choice` — can still differ syntactically, and only because
/// `serde_json` is built with `preserve_order`.
pub fn canonicalize(body: &[u8]) -> Option<Bytes> {
	let mut value: serde_json::Value = serde_json::from_slice(body).ok()?;
	sort_keys(&mut value);
	serde_json::to_vec(&value).ok().map(Bytes::from)
}

fn sort_keys(value: &mut serde_json::Value) {
	match value {
		serde_json::Value::Object(map) => {
			let mut entries: Vec<(String, serde_json::Value)> = std::mem::take(map).into_iter().collect();
			entries.sort_by(|(a, _), (b, _)| a.cmp(b));
			for (_, v) in entries.iter_mut() {
				sort_keys(v);
			}
			*map = entries.into_iter().collect();
		},
		serde_json::Value::Array(items) => {
			for item in items {
				sort_keys(item);
			}
		},
		_ => {},
	}
}

/// A cache consulted for one request, with its key and TTL already resolved.
pub struct PreparedCache {
	pub cache: Arc<ResponseCache>,
	pub key: CacheKey,
	pub ttl: Duration,
}

/// Decide whether a request may use the cache and, if so, build its key.
///
/// `None` is the bypass path and is deliberately the answer to every uncertainty — an ineligible
/// route, a dimension that cannot be represented, a TTL that does not evaluate. The request then
/// proceeds exactly as it would with no cache configured.
pub fn prepare(
	policy: Option<&crate::llm::Policy>,
	llm_request: Option<&crate::llm::LLMRequest>,
	target: &crate::types::agent::Target,
	req: &crate::http::Request,
) -> Option<PreparedCache> {
	let cache = policy?.response_cache.clone()?;
	if !cache.serves_hits() {
		return None;
	}
	let llm_request = llm_request?;
	// First version: chat completions, buffered. Streaming replay is a separate problem and is not
	// attempted here; other route types keep their current behavior untouched.
	if llm_request.input_format != crate::llm::InputFormat::Completions || llm_request.streaming {
		return None;
	}
	let canonical = req.extensions().get::<CanonicalRequestBody>()?;
	let ttl = cache.ttl(req)?;
	let uri = req.uri();
	let target = target.to_string();
	let inputs = CacheKeyInputs {
		canonical_body: &canonical.0,
		provider: llm_request.provider.as_str(),
		model: llm_request.request_model.as_str(),
		input_format: &format!("{:?}", llm_request.input_format),
		target: &target,
		path_and_query: uri.path_and_query().map(|p| p.as_str()).unwrap_or("/"),
		headers: req.headers(),
	};
	let key = cache.key(&inputs, req)?;
	Some(PreparedCache { cache, key, ttl })
}

/// Store a successful upstream response, returning the response either way.
///
/// Only responses declaring a `content-length` within the buffer limit are stored. Collecting an
/// unbounded body here could fail a request that would otherwise have succeeded, and a cache is
/// never allowed to do that; an unmeasurable response is simply not cached.
pub async fn store(
	prepared: &PreparedCache,
	resp: crate::http::Response,
	limit: usize,
) -> Result<crate::http::Response, crate::http::Error> {
	if !resp.status().is_success() {
		return Ok(resp);
	}
	let length = resp
		.headers()
		.get(header::CONTENT_LENGTH)
		.and_then(|v| v.to_str().ok())
		.and_then(|v| v.parse::<usize>().ok());
	let Some(length) = length else {
		return Ok(resp);
	};
	if length > limit {
		return Ok(resp);
	}
	let (parts, body) = resp.into_parts();
	let bytes = http_body_util::BodyExt::collect(body).await?.to_bytes();
	prepared.cache.insert(
		prepared.key,
		CachedResponse::capture(parts.status, &parts.headers, bytes.clone(), prepared.ttl),
	);
	Ok(crate::http::Response::from_parts(
		parts,
		crate::http::Body::from(bytes),
	))
}

#[cfg(test)]
#[path = "response_cache_tests.rs"]
mod tests;
