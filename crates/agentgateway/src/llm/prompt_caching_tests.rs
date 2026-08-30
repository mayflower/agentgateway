//! `promptCaching` applied to the native Anthropic provider.
//!
//! The per-boundary rules live in `agent_llm::conversion::messages_caching`; these tests cover
//! the wiring: which providers the policy reaches, and that both Anthropic input paths get it.

use ::http::HeaderMap;
use agent_llm::{anthropic, vertex};

use super::*;

fn caching() -> policy::PromptCachingConfig {
	policy::PromptCachingConfig {
		cache_system: true,
		cache_messages: true,
		cache_tools: true,
		min_tokens: None,
		cache_message_offset: 0,
	}
}

fn anthropic_provider() -> AIProvider {
	AIProvider::Anthropic(anthropic::Provider { model: None })
}

fn vertex_provider() -> AIProvider {
	AIProvider::Vertex(vertex::Provider {
		model: None,
		region: None,
		project_id: strng::literal!("p"),
	})
}

/// Two-turn conversation so the message boundary (second-to-last) exists.
const MESSAGES_BODY: &str = r#"{
	"model": "claude-sonnet-4",
	"max_tokens": 100,
	"system": "you are helpful",
	"messages": [
		{"role": "user", "content": "first"},
		{"role": "assistant", "content": "reply"},
		{"role": "user", "content": "second"}
	]
}"#;

const COMPLETIONS_BODY: &str = r#"{
	"model": "claude-sonnet-4",
	"messages": [
		{"role": "system", "content": "you are helpful"},
		{"role": "user", "content": "first"},
		{"role": "assistant", "content": "reply"},
		{"role": "user", "content": "second"}
	]
}"#;

fn messages_request() -> types::ChatRequest {
	types::ChatRequest::Messages(serde_json::from_str(MESSAGES_BODY).expect("messages body"))
}

fn completions_request() -> types::ChatRequest {
	types::ChatRequest::Completions(serde_json::from_str(COMPLETIONS_BODY).expect("completions body"))
}

fn render(
	translation: ChatTranslation,
	req: types::ChatRequest,
	provider: &AIProvider,
	caching: Option<&policy::PromptCachingConfig>,
) -> serde_json::Value {
	let headers = HeaderMap::new();
	let rendered = translation
		.render_request(
			req,
			&ChatRequestContext {
				provider,
				headers: &headers,
				prompt_caching: caching,
			},
		)
		.expect("render");
	serde_json::from_slice(&rendered.body).expect("rendered body is json")
}

fn markers(body: &serde_json::Value) -> usize {
	fn walk(v: &serde_json::Value) -> usize {
		match v {
			serde_json::Value::Object(o) => {
				let here = usize::from(o.contains_key("cache_control"));
				here + o.values().map(walk).sum::<usize>()
			},
			serde_json::Value::Array(a) => a.iter().map(walk).sum(),
			_ => 0,
		}
	}
	walk(body)
}

// ── native Anthropic gets policy-generated markers ──────────────────────────

#[test]
fn native_messages_input_gets_markers() {
	let t = ChatTranslation {
		input: InputFormat::Messages,
		output: ChatFormat::AnthropicMessages,
	};
	let body = render(
		t,
		messages_request(),
		&anthropic_provider(),
		Some(&caching()),
	);

	assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
	// Default boundary is the second-to-last message.
	assert_eq!(
		body["messages"][1]["content"][0]["cache_control"]["type"],
		"ephemeral"
	);
}

#[test]
fn completions_translated_to_anthropic_gets_markers() {
	let t = ChatTranslation {
		input: InputFormat::Completions,
		output: ChatFormat::AnthropicMessages,
	};
	let body = render(
		t,
		completions_request(),
		&anthropic_provider(),
		Some(&caching()),
	);

	assert!(
		markers(&body) > 0,
		"completions-to-anthropic should receive generated markers: {body}"
	);
	assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
}

// ── no policy is a byte-for-byte no-op ──────────────────────────────────────

#[test]
fn no_policy_leaves_both_paths_unchanged() {
	for (input, req) in [
		(InputFormat::Messages, messages_request()),
		(InputFormat::Completions, completions_request()),
	] {
		let t = ChatTranslation {
			input,
			output: ChatFormat::AnthropicMessages,
		};
		let body = render(t, req, &anthropic_provider(), None);
		assert_eq!(
			markers(&body),
			0,
			"no policy must not introduce markers: {body}"
		);
	}
}

// ── provider isolation ──────────────────────────────────────────────────────

#[test]
fn vertex_anthropic_is_unchanged() {
	// Vertex renders the same AnthropicMessages format, so gating must be on the provider.
	let t = || ChatTranslation {
		input: InputFormat::Messages,
		output: ChatFormat::AnthropicMessages,
	};
	let with = render(
		t(),
		messages_request(),
		&vertex_provider(),
		Some(&caching()),
	);
	let without = render(t(), messages_request(), &vertex_provider(), None);

	assert_eq!(markers(&with), 0, "vertex must not receive markers");
	assert_eq!(with, without, "policy must not alter vertex output at all");
}

#[test]
fn openai_output_is_unchanged() {
	let t = || ChatTranslation {
		input: InputFormat::Completions,
		output: ChatFormat::OpenAICompletions,
	};
	let provider = AIProvider::OpenAI(agent_llm::openai::Provider {
		model: None,
		moderation: None,
	});
	let with = render(t(), completions_request(), &provider, Some(&caching()));
	let without = render(t(), completions_request(), &provider, None);
	assert_eq!(with, without);
}

// ── explicit client markers ─────────────────────────────────────────────────

#[test]
fn explicit_marker_survives_and_coexists_with_generated() {
	let body_with_marker = r#"{
		"model": "claude-sonnet-4",
		"max_tokens": 100,
		"system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
		"messages": [
			{"role": "user", "content": "first"},
			{"role": "assistant", "content": "reply"},
			{"role": "user", "content": "second"}
		]
	}"#;
	let t = ChatTranslation {
		input: InputFormat::Messages,
		output: ChatFormat::AnthropicMessages,
	};
	let req = types::ChatRequest::Messages(serde_json::from_str(body_with_marker).expect("body"));
	let body = render(t, req, &anthropic_provider(), Some(&caching()));

	// The explicit marker keeps its TTL rather than being replaced by the default form.
	assert_eq!(
		body["system"][0]["cache_control"],
		serde_json::json!({"type": "ephemeral", "ttl": "1h"})
	);
	// And the message boundary still gets its generated marker.
	assert_eq!(
		body["messages"][1]["content"][0]["cache_control"]["type"],
		"ephemeral"
	);
}

#[test]
fn openai_prompt_cache_breakpoint_still_translates_and_coexists() {
	// #3109 translates an explicit OpenAI breakpoint into an Anthropic cache_control. That
	// translation must survive, and consume budget before any generated marker.
	let body = r#"{
		"model": "claude-sonnet-4",
		"messages": [
			{"role": "system", "content": "you are helpful"},
			{"role": "user", "content": [
				{"type": "text", "text": "first", "prompt_cache_breakpoint": {"mode": "explicit"}}
			]},
			{"role": "assistant", "content": "reply"},
			{"role": "user", "content": "second"}
		]
	}"#;
	let t = ChatTranslation {
		input: InputFormat::Completions,
		output: ChatFormat::AnthropicMessages,
	};
	let req = types::ChatRequest::Completions(serde_json::from_str(body).expect("body"));
	let translated = render(t, req, &anthropic_provider(), Some(&caching()));

	// The translated client marker is present ...
	assert_eq!(
		translated["messages"][0]["content"][0]["cache_control"]["type"], "ephemeral",
		"translated breakpoint must survive: {translated}"
	);
	// ... alongside the generated system marker, and within Anthropic's budget.
	assert_eq!(
		translated["system"][0]["cache_control"]["type"],
		"ephemeral"
	);
	assert!(
		markers(&translated) <= 4,
		"must not exceed the breakpoint budget: {translated}"
	);
}
