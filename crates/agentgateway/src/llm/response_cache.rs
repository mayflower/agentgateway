//! Gateway-side exact response cache for buffered chat completions. A hit replaces the provider
//! call with a stored response for an identical finalized request. This is not provider-side prompt
//! caching: no marker is involved and nothing sent upstream changes.
//!
//! The CEL `key` and `ttl` follow `http::ext_authz::CacheConfig`; the length-delimited SHA-256
//! digest follows the OAuth token cache, so no raw prompt or credential is retained as key
//! material.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ::http::{HeaderMap, HeaderName, StatusCode, header};
use bytes::Bytes;
use quick_cache::sync::Cache;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::crypto::digest::Sha256;
use crate::*;

const DEFAULT_MAX_ENTRIES: usize = 1024;

fn default_max_entries() -> usize {
	DEFAULT_MAX_ENTRIES
}

/// Headers that change the upstream result without appearing in the body. `anthropic-beta` is
/// forwarded verbatim on the native Anthropic path, though Bedrock folds it into the body instead.
/// `accept-encoding` is here for a different reason: it decides how the stored bytes are encoded,
/// so without it a gzip response could be replayed to a client that never asked for gzip.
/// `anthropic-version` is gateway-set and Azure's `api-version` is in the query, so neither varies.
const KEYED_REQUEST_HEADERS: &[&str] = &[
	"anthropic-beta",
	"openai-organization",
	"openai-project",
	"accept-encoding",
];

/// The credential the gateway attached on the way out, as opposed to the caller the operator's
/// `key` expressions name. Two routes can share one backend policy, and so one cache, while
/// attaching different credentials; a stored response must not cross that boundary unnoticed.
const CREDENTIAL_HEADERS: &[&str] = &["authorization", "x-api-key", "api-key"];

/// Response headers replayed from a stored entry. An allowlist: `date`, `x-request-id`,
/// `set-cookie`, and the provider rate-limit headers all describe an exchange the current client did
/// not make. `content-encoding` qualifies because it describes the stored bytes, not the exchange.
const REPLAYED_RESPONSE_HEADERS: &[&str] = &["content-type", "content-encoding"];

#[apply(schema!)]
pub struct ResponseCacheConfig {
	/// Extra cache-key dimensions, evaluated against the request. The gateway always keys on the
	/// finalized body, provider, model, format, upstream target and path, and the attached credential;
	/// these add what only the deployment knows, such as the caller a response must not be shared
	/// across. With no expressions the cache serves no hits, rather than sharing silently.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub key: Vec<Arc<cel::Expression>>,
	/// How long a stored response may be reused. Accepts a duration literal such as `5m`, or a CEL
	/// expression returning a duration.
	#[serde(deserialize_with = "crate::cel::de_duration_or_expression")]
	pub ttl: Arc<cel::Expression>,
	/// Maximum number of stored responses; `0` disables storage. Bounds the entry count, not bytes:
	/// the worst case is `maxEntries` times the route's response buffer limit. Per-process, and lost
	/// on restart.
	#[serde(default = "default_max_entries")]
	pub max_entries: usize,
}

/// Held behind an `Arc` on the LLM policy so per-request policy merging shares one store, and so a
/// configuration reload builds a new policy with an empty one. That is the whole invalidation story
/// for a config change; there is no imperative invalidation API.
pub struct ResponseCache {
	config: ResponseCacheConfig,
	store: Cache<CacheKey, CachedResponse>,
}

impl ResponseCache {
	pub fn new(config: ResponseCacheConfig) -> Self {
		// `quick_cache` panics on zero capacity; a configured zero means storage is disabled, which
		// `serves_hits` already reports.
		let capacity = config.max_entries.max(1);
		Self {
			store: Cache::new(capacity),
			config,
		}
	}

	pub fn expressions(&self) -> impl Iterator<Item = &cel::Expression> {
		self
			.config
			.key
			.iter()
			.map(Arc::as_ref)
			.chain(std::iter::once(self.config.ttl.as_ref()))
	}

	/// An empty key list cannot express caller identity, so a hit would cross callers.
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

	/// A TTL that does not evaluate to a positive duration disables storage for this request, rather
	/// than falling back to a default the operator did not choose.
	pub fn ttl(&self, req: &crate::http::Request) -> Option<Duration> {
		let exec = cel::Executor::new_request(req);
		let value = exec.eval(&self.config.ttl).ok()?;
		let cel::Value::Duration(ttl) = value else {
			return None;
		};
		let ttl = ttl.to_std().ok()?;
		(!ttl.is_zero()).then_some(ttl)
	}

	/// `None` when a dimension cannot be represented: that bypasses, never a partial key.
	pub fn key(&self, inputs: &CacheKeyInputs<'_>, req: &crate::http::Request) -> Option<CacheKey> {
		let mut digest = CacheKeyDigest::new();

		// Gateway-computed dimensions the operator cannot omit.
		digest.field(inputs.canonical_body);
		digest.field(inputs.provider.as_bytes());
		digest.field(inputs.model.as_bytes());
		digest.field(inputs.input_format.as_bytes());
		digest.field(inputs.target.as_bytes());
		digest.field(inputs.path_and_query.as_bytes());

		// Iterated over the fixed list so the digest does not depend on header map ordering, and no
		// un-allowlisted header can silently join the key.
		for name in KEYED_REQUEST_HEADERS {
			digest.field(name.as_bytes());
			for value in inputs.headers.get_all(*name) {
				digest.field(value.as_bytes());
			}
		}

		// A credential re-derived per request can neither be keyed on, since the key would never match
		// twice, nor left out. Such a request bypasses: narrower eligibility over a partial key.
		for name in CREDENTIAL_HEADERS {
			digest.field(name.as_bytes());
			for value in inputs.headers.get_all(*name) {
				if !credential_is_stable(value.as_bytes()) {
					return None;
				}
				digest.field(value.as_bytes());
			}
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

/// SigV4 rebuilds `Authorization` per request from a timestamp and signature, identifying the
/// exchange rather than the principal. Extracting the stable credential scope would mean parsing it.
fn credential_is_stable(value: &[u8]) -> bool {
	!value.starts_with(b"AWS4-")
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

pub struct CacheKeyInputs<'a> {
	pub canonical_body: &'a [u8],
	pub provider: &'a str,
	pub model: &'a str,
	pub input_format: &'a str,
	/// The upstream wire format is not a separate field: the provider and its path already imply it.
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

/// Length-delimited so field boundaries cannot be forged by concatenating adjacent values.
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

/// Held in the shape it arrived in, so replay re-enters the normal buffered path: translation,
/// response guards, usage accounting, and logging all run as they would on a miss.
#[derive(Clone, Debug)]
pub struct CachedResponse {
	status: StatusCode,
	headers: HeaderMap,
	body: Bytes,
	expires_at: Instant,
}

impl CachedResponse {
	/// Keeps only the allowlisted headers.
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

/// The finalized upstream body, carried from rendering to where the key is built. Canonicalized on
/// insertion rather than at key time, which keeps JSON handling in the LLM layer.
#[derive(Clone, Debug)]
pub struct CanonicalRequestBody(pub Bytes);

/// Recursively order object keys, leaving arrays, values, and prompt text untouched. That is the
/// whole surface: the request was already parsed and re-serialized with `serde_json::to_vec`, so
/// only raw-JSON values such as the flattened `rest` maps and `tools` can still differ, and only
/// because `serde_json` is built with `preserve_order`.
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

pub struct PreparedCache {
	pub cache: Arc<ResponseCache>,
	pub key: CacheKey,
	pub ttl: Duration,
}

/// `None` is the bypass path and the answer to every uncertainty: an ineligible route, a dimension
/// that cannot be represented, a TTL that does not evaluate. The request then proceeds as it would
/// with no cache configured.
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
	// First version: buffered chat completions. Streaming replay is a separate problem.
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

/// Only responses declaring a `content-length` within the buffer limit are stored. Collecting an
/// unbounded body here could fail a request that would otherwise have succeeded.
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
