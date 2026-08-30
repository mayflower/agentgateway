//! Policy-generated Anthropic `cache_control` markers.
//!
//! The `promptCaching` policy already generates `cachePoint` blocks for Bedrock during request
//! translation (see `conversion::bedrock`). This module does the same for the native Anthropic
//! Messages API.
//!
//! Anthropic is reached through two different request representations — native Messages input
//! (`types::messages::Request`) and OpenAI Completions translated to Messages
//! (`types::messages::typed::Request`) — but both serialize to the same Anthropic wire shape.
//! Applying the policy to that shape keeps one implementation instead of two, and passes
//! unknown fields through untouched because they are still just JSON.
//!
//! Boundary and offset semantics deliberately mirror the Bedrock implementation so a single
//! `promptCaching` policy behaves the same on both providers.

use serde_json::{Map, Value, json};

use crate::PromptCachingConfig;

/// Anthropic rejects requests carrying more than four cache breakpoints.
const MAX_CACHE_BREAKPOINTS: usize = 4;

/// Apply the policy to a request already in Anthropic Messages wire shape.
///
/// Best-effort: a request with no cacheable boundary, or no remaining breakpoint budget, is
/// left as-is rather than rejected. Client-supplied markers are never modified or removed.
pub fn apply(req: &mut Value, caching: &PromptCachingConfig) {
	let Some(req) = req.as_object_mut() else {
		return;
	};

	// Client-supplied markers consume the budget first; we only ever use what is left.
	let mut used = count_markers(req);

	// Order matches the Bedrock implementation: system, then messages, then tools.
	if caching.cache_system && used < MAX_CACHE_BREAKPOINTS && apply_system(req, caching) {
		used += 1;
	}
	if caching.cache_messages && used < MAX_CACHE_BREAKPOINTS && apply_messages(req, caching) {
		used += 1;
	}
	if caching.cache_tools && used < MAX_CACHE_BREAKPOINTS && apply_tools(req) {
		used += 1;
	}
	debug_assert!(used <= MAX_CACHE_BREAKPOINTS);
}

/// Count `cache_control` markers already present anywhere in the request.
fn count_markers(req: &Map<String, Value>) -> usize {
	let system = req.get("system").map(count_in_blocks).unwrap_or(0);
	let tools = req.get("tools").map(count_in_blocks).unwrap_or(0);
	let messages = req
		.get("messages")
		.and_then(Value::as_array)
		.map(|messages| {
			messages
				.iter()
				.map(|m| m.get("content").map(count_in_blocks).unwrap_or(0))
				.sum()
		})
		.unwrap_or(0);
	system + messages + tools
}

/// Count markers in a value that is either a block array or a bare string (never marked).
fn count_in_blocks(value: &Value) -> usize {
	value
		.as_array()
		.map(|blocks| blocks.iter().filter(|b| has_marker(b)).count())
		.unwrap_or(0)
}

fn has_marker(block: &Value) -> bool {
	block
		.get("cache_control")
		.is_some_and(|marker| !marker.is_null())
}

/// Add the default ephemeral marker. Returns false when the block already carries one, so an
/// explicit client marker (including its TTL or other fields) is never overwritten.
fn mark(block: &mut Value) -> bool {
	let Some(block) = block.as_object_mut() else {
		return false;
	};
	if block
		.get("cache_control")
		.is_some_and(|marker| !marker.is_null())
	{
		return false;
	}
	block.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
	true
}

/// Coerce Anthropic's string shorthand into the equivalent single-text-block array so it can
/// carry a marker. Anything already in block form is left alone.
fn as_blocks(value: &mut Value) -> Option<&mut Vec<Value>> {
	if let Some(text) = value.as_str() {
		*value = json!([{"type": "text", "text": text}]);
	}
	value.as_array_mut()
}

/// Content block types that accept `cache_control`, mirroring the blocks that carry the field in
/// `types::messages::typed::ContentBlock`. `thinking` and `redacted_thinking` deliberately do not.
///
/// An unrecognised type is skipped rather than marked: losing an optimization is harmless, while
/// attaching a marker Anthropic rejects would break an otherwise valid request.
const CACHEABLE_BLOCK_TYPES: &[&str] = &[
	"text",
	"image",
	"document",
	"search_result",
	"tool_use",
	"tool_result",
	"server_tool_use",
	"web_search_tool_result",
];

fn is_cacheable_block(block: &Value) -> bool {
	block
		.get("type")
		.and_then(Value::as_str)
		.is_some_and(|t| CACHEABLE_BLOCK_TYPES.contains(&t))
}

/// Mark the last content block that can carry a marker, walking back past blocks that cannot
/// (a trailing `thinking` block, or a non-object entry).
fn mark_last_content_block(blocks: &mut [Value]) -> bool {
	blocks
		.iter_mut()
		.rfind(|block| is_cacheable_block(block))
		.is_some_and(mark)
}

/// Mark the last tool. Tool definitions all accept `cache_control`, and custom tools carry no
/// `type` discriminator, so the content-block allowlist does not apply here.
fn mark_last_tool(tools: &mut [Value]) -> bool {
	tools
		.iter_mut()
		.rfind(|tool| tool.is_object())
		.is_some_and(mark)
}

fn apply_system(req: &mut Map<String, Value>, caching: &PromptCachingConfig) -> bool {
	let Some(system) = req.get_mut("system") else {
		return false;
	};
	// `minTokens` gates the system boundary only, matching the Bedrock implementation.
	if let Some(min_tokens) = caching.min_tokens
		&& estimate_system_tokens(system) < min_tokens
	{
		return false;
	}
	as_blocks(system).is_some_and(|blocks| mark_last_content_block(blocks))
}

fn apply_messages(req: &mut Map<String, Value>, caching: &PromptCachingConfig) -> bool {
	let Some(messages) = req.get_mut("messages").and_then(Value::as_array_mut) else {
		return false;
	};
	// Cache the conversation history but not the current turn: the default boundary is the
	// second-to-last message, and `cacheMessageOffset` walks it further back, clamped at the
	// first message. Nothing to reuse yet with fewer than two messages.
	if messages.len() < 2 {
		return false;
	}
	let target = (messages.len() - 2).saturating_sub(caching.cache_message_offset);
	let Some(content) = messages[target].get_mut("content") else {
		return false;
	};
	as_blocks(content).is_some_and(|blocks| mark_last_content_block(blocks))
}

fn apply_tools(req: &mut Map<String, Value>) -> bool {
	req
		.get_mut("tools")
		.and_then(Value::as_array_mut)
		.is_some_and(|tools| mark_last_tool(tools))
}

/// Rough token estimate for the system prompt, matching the Bedrock heuristic
/// (whitespace-separated words, scaled by 1.3) so `minTokens` means the same thing on both.
fn estimate_system_tokens(system: &Value) -> usize {
	let words = match system {
		Value::String(text) => text.split_whitespace().count(),
		Value::Array(blocks) => blocks
			.iter()
			.filter_map(|block| block.get("text").and_then(Value::as_str))
			.map(|text| text.split_whitespace().count())
			.sum(),
		_ => 0,
	};
	(words * 13) / 10
}

#[cfg(test)]
#[path = "messages_caching_tests.rs"]
mod tests;
